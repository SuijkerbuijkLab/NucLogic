"""CPU nuclear segmentation with a size-calibrated watershed, one 2D label image per Z plane.

This is a lightweight alternative to Cellpose for computers without a GPU. Like the
Cellpose path, it segments every Z plane in 2D; NucLogic's stitching
(`utils/stitch_3d.py`) then links the 2D masks into 3D nuclei.

The main difficulty is that image quality changes with depth: deep planes are dim and
noisy, shallow planes are clear. So the algorithm learns the nucleus size once, on the
clearest planes, and expresses every threshold in units of each plane's own noise.

How a stack is segmented (`WatershedSegmenter.segment_stack`):

1. Light blur along Z, so every plane borrows some signal from its neighbours.
2. Per plane: estimate the noise level and a quality score (how much bright structure
   stands out from that noise).
3. Calibration: on the few best planes, estimate the nucleus radius R. A first guess
   comes from a blob detector at many scales; then we segment, measure the objects that
   do not touch anything, and repeat until R stops changing.
4. Per plane (in parallel threads):
   a. work at a reduced resolution where nuclei are ~6 px in radius (much faster),
   b. foreground = smoothed image above a noise-based threshold,
   c. seeds = blob-detector peaks matched to size R (one per nucleus),
   d. watershed from the seeds to split touching nuclei,
   e. per object: put the edge at half its height, split objects with a 'neck'
      (two nuclei that merged), and drop haze that barely stands out from its
      surroundings,
   f. drop objects that are not clearly above the camera noise; this keeps empty
      planes (above or below the sample) empty.

Parameters worth tuning first: `diameter` (skip calibration if you know the size),
`k_object` (higher = stricter about faint objects), `min_rel_contrast` (higher = more
aggressive against haze), `neck_depth` (lower = split touching nuclei more readily).

A few terms used throughout:
- sigma / noise sigma: the typical size of random pixel-to-pixel fluctuations.
  "k sigmas" means "k times larger than typical noise".
- LoG (Laplacian of Gaussian): a filter that responds strongly to round bright blobs of
  a particular size; used here as the blob/nucleus detector.
- MAD (median absolute deviation): a measure of spread like the standard deviation, but
  hardly affected by a few very bright pixels (such as nuclei). 1.4826 * MAD equals the
  standard deviation for normally distributed noise.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
from alive_progress import alive_bar
from scipy import ndimage as ndi
from skimage import morphology, segmentation
from skimage.feature import peak_local_max


# The linear filters used in this file. We keep them in one place because we need to know
# how much each one shrinks the noise (see WatershedSegmenter._norm).
FILTERS = {
    "residual": lambda x, s: x - ndi.gaussian_filter(x, s),  # image minus a blurred copy: keeps fine detail/noise
    "gauss": ndi.gaussian_filter,                           # blur
    "log": ndi.gaussian_laplace,                            # blob detector (negative for bright blobs)
}


class WatershedSegmenter:
    """CPU 2D nuclear segmentation of a 3D stack, one label image per Z plane.

    Usage:
        segmenter = WatershedSegmenter()
        labels = segmenter.segment_stack(image)   # image: (Z, Y, X) array -> labels: (Z, Y, X) uint16
        segmenter.radius                          # the nucleus radius it found (pixels)
        segmenter.last                            # per-plane diagnostics (noise, quality, seeds, ...)

    The nucleus radius R is measured once on the highest-quality planes and then sets the
    blur, the seed detector and the seed spacing on every plane. Thresholds are in units of
    each plane's own noise, so planes without signal come out empty instead of segmenting
    stretched noise.
    """

    def __init__(self, diameter=None, z_sigma=1.0, k_fg=3.0, rel_fg=0.25, k_seed=3.0,
                 top_k=3, quality_floor=4.0, radius_percentile=75, compactness=0.0,
                 edge_frac=0.5, neck_depth=0.2, calibration_passes=3, min_rel_contrast=0.25,
                 target_radius=6.0, disk_like=0.8, k_object=12.0, workers=None):
        # Size
        self.diameter = diameter            # nucleus diameter in px; None = measure it on the best planes
        self.top_k = top_k                  # number of best planes used to measure the size
        self.calibration_passes = calibration_passes  # segment-and-remeasure rounds for R
        self.radius_percentile = radius_percentile    # R = this percentile of measured radii (75: full-size nuclei, not small cross-sections)
        self.target_radius = target_radius  # working scale: planes are downsampled until nuclei are about this radius

        # Detection
        self.z_sigma = z_sigma              # blur along Z, in planes; 0 = off
        self.k_fg = k_fg                    # foreground threshold in noise sigmas...
        self.rel_fg = rel_fg                # ...or as a fraction of the bright-nucleus level, whichever is stricter
        self.k_seed = k_seed                # seed (blob peak) threshold in noise sigmas
        self.quality_floor = quality_floor  # planes scoring below this are skipped; pure noise scores about 2.3
        self.compactness = compactness      # >0 makes watershed regions rounder (0 = follow the intensity only)

        # Clean-up of each object
        self.edge_frac = edge_frac          # object edge at this fraction of its height (0.5 = half maximum)
        self.neck_depth = neck_depth        # split if the neck is this much narrower than the lobes, in units of R
        self.disk_like = disk_like          # deepest point / equivalent radius; two touching disks reach at most 0.71
        self.min_rel_contrast = min_rel_contrast  # drop objects whose signal is mostly a local pedestal (haze); 0 = off
        self.k_object = k_object            # objects must peak this many camera-noise sigmas out; 0 = off

        # Speed
        self.workers = workers or min(8, os.cpu_count() or 1)  # threads; planes are processed in parallel

        # Filled in while running
        self.radius = None                  # nucleus radius in px (full resolution)
        self.last = {}                      # diagnostics of the last segment_stack call
        self._norms = {}                    # cache for _norm

    # =====================================================================================
    # Small helpers
    # =====================================================================================

    def _map(self, fn, items):
        """Run fn on every item, using a pool of threads.

        Threads help here because nearly all heavy work happens inside SciPy/scikit-image C
        code, which releases Python's GIL (the lock that normally lets only one thread run
        Python at a time). Beyond ~8 threads there is little gain: memory speed becomes the limit.
        """
        if self.workers <= 1:
            return [fn(i) for i in items]
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(fn, items))

    def _norm(self, kind, sigma):
        """How much a filter scales random noise: noise after filtering = noise before * this.

        We need this to express thresholds in 'noise sigmas' on filtered images. It is
        measured by filtering a single bright pixel (the filter's 'kernel') and taking the
        square root of the sum of squares. Results are cached because the same filters are
        used on every plane.
        """
        key = (kind, round(float(sigma), 3))
        if key not in self._norms:
            size = 2 * int(4 * sigma + 2) + 1
            delta = np.zeros((size, size), np.float32)
            delta[size // 2, size // 2] = 1
            self._norms[key] = float(np.sqrt(np.sum(FILTERS[kind](delta, sigma) ** 2)))
        return self._norms[key]

    @staticmethod
    def block_mean(plane, f):
        """Shrink an image f times by averaging every f x f block of pixels.

        Averaging (rather than skipping pixels) also averages out noise. Rows/columns that
        do not fill a whole block are dropped.
        """
        if f == 1:
            return plane
        h, w = (plane.shape[0] // f) * f, (plane.shape[1] // f) * f
        out = np.zeros((h // f, w // f), np.float32)
        for i in range(f):  # summing strided views is ~3x faster than reshape().mean()
            for j in range(f):
                out += plane[i:h:f, j:w:f]
        return out / (f * f)

    @staticmethod
    def _resize(m, shape):
        """Resize a 2D map to `shape` with linear interpolation (used by pyramid_radius)."""
        out = ndi.zoom(m, np.array(shape) / np.array(m.shape), order=1, grid_mode=True, mode="nearest")
        out = out[: shape[0], : shape[1]]
        return np.pad(out, [(0, shape[0] - out.shape[0]), (0, shape[1] - out.shape[1])], mode="edge")

    @staticmethod
    def _label_reduce(values, labels, n, op):
        """Per object: the maximum (op=np.maximum) or minimum (op=np.minimum) of `values`.

        Returns an array indexed by label (index 0 = background); labels that do not occur
        get NaN. Equivalent to scipy's ndi.maximum/ndi.minimum, but much faster because it
        only sorts the labelled pixels instead of the whole image.
        """
        m = labels > 0
        lab, val = labels[m], values[m]
        # Sort the pixels by label so each object's pixels are consecutive...
        order = np.argsort(lab, kind="stable")
        lab, val = lab[order], val[order]
        out = np.full(n + 1, np.nan)
        if lab.size:
            # ...then reduce each consecutive run in one call.
            starts = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
            out[lab[starts]] = op.reduceat(val, starts)
        return out

    # =====================================================================================
    # Per-plane measurements: how noisy is a plane, and how good is it?
    # =====================================================================================

    def noise_sigma(self, plane, valid):
        """Estimate how noisy this plane is (the noise 'sigma').

        Almost all thresholds in this file are 'k times the noise', so this number
        decides what counts as signal.

        Trick: blur the plane slightly and subtract the blur from the original. Nuclei are
        smooth, so they mostly cancel out; pixel noise is not smooth, so it remains. The
        spread (MAD) of what is left is the noise.
        """
        # Stay 2 px away from clipped (zero) regions: the jump to zero would look like noise.
        inner = ndi.binary_erosion(valid, iterations=2)
        if inner.sum() < 100:
            return 0.0
        residual = FILTERS["residual"](plane, 1.0)[inner]
        mad = np.median(np.abs(residual - np.median(residual)))
        # The blur-and-subtract step also shrinks the noise a little; _norm undoes that.
        return 1.4826 * mad / self._norm("residual", 1.0)

    def quality(self, plane, sigma_n, valid):
        """Score how much bright structure stands out from the noise in this plane.

        Score = (brightest 1% - median) / noise, after a light blur. Pure noise scores about
        2.3; planes with clear nuclei score tens to hundreds. It does not need to know the
        nucleus size, so it can be used to pick the calibration planes.
        """
        if sigma_n <= 0 or not valid.any():
            return 0.0
        # Computed on a 2x smaller copy: same result as a sigma-2 blur at full size, 4x cheaper.
        smooth = ndi.gaussian_filter(self.block_mean(plane, 2), 1.0)
        v = valid[: smooth.shape[0] * 2 : 2, : smooth.shape[1] * 2 : 2]
        if not v.any():
            return 0.0
        p50, p99 = np.percentile(smooth[v], [50, 99])
        # Averaging 2x2 blocks halves the noise, hence sigma_n / 2.
        return (p99 - p50) / (sigma_n / 2 * self._norm("gauss", 1.0))

    def blob_noise(self, image, valid):
        """Per plane: the camera noise level as seen by the blob detector.

        Used for the final 'is this object real?' test (k_object). Measured on the
        difference between a plane and its neighbour: nuclei and haze look almost the same
        in neighbouring planes and cancel out, but camera noise is different in every plane
        and remains. This gives an honest noise level even when the noise has stripes or is
        clipped at zero, where noise_sigma underestimates it.
        """
        f = max(1, int(round(self.radius / self.target_radius)))  # same working scale as segment_plane
        s_log = self.radius / f / np.sqrt(2)
        # The Z blur (z_sigma) also reduces the noise; compute by how much.
        wz = np.exp(-0.5 * (np.arange(-4, 5) / max(self.z_sigma, 1e-3)) ** 2)
        zblur = np.sqrt(np.sum((wz / wz.sum()) ** 2)) if self.z_sigma > 0 else 1.0
        if len(image) < 2:
            return np.zeros(len(image))  # a single plane has no neighbour: test is skipped

        def one(z):
            nb = z + 1 if z + 1 < len(image) else z - 1
            d = self.block_mean(image[nb].astype(np.float32), f) - self.block_mean(image[z].astype(np.float32), f)
            v = valid[z][: d.shape[0] * f : f, : d.shape[1] * f : f] | valid[nb][: d.shape[0] * f : f, : d.shape[1] * f : f]
            if v.sum() < 100:
                return 0.0
            x = ndi.gaussian_laplace(d, s_log)[v]
            # A difference of two planes has sqrt(2) times the noise of one plane.
            return 1.4826 * np.median(np.abs(x - np.median(x))) / np.sqrt(2) * zblur

        return np.array(self._map(one, range(len(image))))

    # =====================================================================================
    # Calibration: find the nucleus radius R
    # =====================================================================================

    def pyramid_radius(self, plane, k=5.0, octaves=(1, 2, 3, 4), per_octave=6):
        """First guess of the nucleus radius, without any intensity threshold.

        A blob detector (LoG) responds most strongly when its size matches the blob. We try
        many sizes and let every clear blob 'vote' for the size it matched best; the median
        vote is the guess. Larger sizes are tried on smaller copies of the plane (an
        'image pyramid'), which is fast and also averages away pixel noise, so noise specks
        cannot win. Covers radii of about 3.5-50 px.

        Octave o works on 2^o x 2^o block means with sigma in [1.2, 2.4), i.e. radius
        2^o * sqrt(2) * sigma in full-resolution pixels.
        """
        ref_shape = self.block_mean(plane, 2 ** octaves[0]).shape
        resp, zsc, radius_of = [], [], []
        for o in octaves:
            f = 2 ** o
            small = self.block_mean(plane, f)
            if min(small.shape) < 16:
                break
            valid = small > 0
            if valid.mean() > 0.99:
                valid[:] = True
            sigma_n = self.noise_sigma(small, valid)
            if sigma_n <= 0:
                continue
            for sg in 1.2 * 2 ** (np.arange(per_octave) / per_octave):
                log = -ndi.gaussian_laplace(small, sg)
                # resp: response made comparable between sizes (to pick the best size);
                # zsc: the same response in noise sigmas (to judge if a blob is real).
                maps = [log * sg**2, log / (sigma_n * self._norm("log", sg))]
                if small.shape != ref_shape:
                    maps = [self._resize(m, ref_shape) for m in maps]
                resp.append(maps[0])
                zsc.append(maps[1])
                radius_of.append(f * np.sqrt(2) * sg)
        if len(radius_of) < 3:
            return np.nan
        resp, zsc, radius_of = np.stack(resp), np.stack(zsc), np.array(radius_of)
        best = resp.argmax(0)  # per pixel: which size matched best
        best_z = np.take_along_axis(zsc, best[None], 0)[0]
        # Blob centres = local maxima; only significant pixels can hold one (also keeps
        # flat zero regions out, which keeps this fast).
        peaks = peak_local_max(resp.max(0), min_distance=2, exclude_border=False,
                               labels=(best_z > k).astype(np.int32))
        b = best[tuple(peaks.T)]
        peaks = peaks[(b > 0) & (b < len(radius_of) - 1)]  # not stuck at the smallest/largest size tried
        if len(peaks) == 0:
            return np.nan
        z = best_z[tuple(peaks.T)]
        strong = peaks[z >= np.median(z)]  # vote with the most significant half
        return float(np.median(radius_of[best[tuple(strong.T)]]))

    @staticmethod
    def isolated_radii(labels):
        """Radius sqrt(area / pi) of every object that touches no other object.

        Why only isolated objects: if R is too small the segmentation over-splits nuclei
        into touching fragments, and measuring those fragments would 'confirm' the small R.
        Fragments always touch their siblings, so skipping touching objects breaks that loop.
        """
        n = int(labels.max())
        if n == 0:
            return np.empty(0)
        # A pixel touches another object if a neighbour has a larger label (grey dilation)
        # or a smaller one (grey erosion, with background replaced by a large value).
        grown = ndi.grey_dilation(labels, size=3)
        shrunk = ndi.grey_erosion(np.where(labels > 0, labels, n + 1), size=3)
        contact = (labels > 0) & ((grown != labels) | (shrunk != labels))
        touching = np.zeros(n + 1, bool)
        touching[labels[contact]] = True
        keep = np.flatnonzero(~touching[1:]) + 1
        return np.sqrt(np.bincount(labels.ravel(), minlength=n + 1)[keep] / np.pi)

    def calibrate_radius(self, planes, noises, valids):
        """Find the nucleus radius R on a few good planes.

        Start from the pyramid_radius guess, then repeat: segment with the current R,
        measure the isolated objects, set R to the radius_percentile of those radii. Stops
        when R changes less than 5% (usually after 2-3 rounds).
        """
        guesses = self._map(self.pyramid_radius, planes)
        guess = np.nanmedian(guesses) if np.isfinite(guesses).any() else 5.0
        self.radius = float(np.clip(guess, 2, 50))
        history, radii = [self.radius], np.empty(0)
        for _ in range(self.calibration_passes):
            radii = np.concatenate(self._map(lambda k: self.isolated_radii(self.segment_plane(planes[k], noises[k], valids[k])[0]),
                                             range(len(planes))))
            if radii.size < 10:
                break
            new = float(np.clip(np.percentile(radii, self.radius_percentile), 2, 50))
            history.append(new)
            converged = abs(new - self.radius) < 0.05 * self.radius
            self.radius = new
            if converged:
                break
        return radii, history

    # =====================================================================================
    # Segmenting one plane
    # =====================================================================================

    def segment_plane(self, plane, sigma_n, valid, blob_noise=None):
        """Segment one plane; returns (labels, seed coordinates) at full resolution.

        Large nuclei do not need full resolution, so the plane is first shrunk until nuclei
        are about target_radius px (e.g. R = 12 -> 2x smaller, 4x fewer pixels). The result
        is scaled back up afterwards.
        """
        f = max(1, int(round(self.radius / self.target_radius)))
        if f == 1:
            return self._segment_plane(plane, sigma_n, valid, self.radius, blob_noise)
        small = self.block_mean(plane, f)
        valid_small = valid[: small.shape[0] * f : f, : small.shape[1] * f : f]
        # Averaging f x f blocks divides the noise by f.
        labels, seeds = self._segment_plane(small, sigma_n / f, valid_small, self.radius / f, blob_noise)
        full = np.zeros(plane.shape, np.uint16)
        full[: labels.shape[0] * f, : labels.shape[1] * f] = labels.repeat(f, 0).repeat(f, 1)
        return full, seeds * f + f // 2

    def _segment_plane(self, plane, sigma_n, valid, R, blob_noise=None):
        """The actual per-plane segmentation, at the working scale (R in working-scale px)."""
        no_seeds = np.empty((0, 2), int)

        # 1. Foreground: which pixels belong to some nucleus?
        #    Blur at R/3 (removes noise, keeps nuclei), then threshold. The threshold is the
        #    background plus whichever is stricter: k_fg noise sigmas, or rel_fg of the way
        #    to the bright nuclei. The first protects noisy planes, the second clear ones.
        s_smooth = max(1.0, R / 3)
        smooth = ndi.gaussian_filter(plane, s_smooth)
        noise_smooth = sigma_n * self._norm("gauss", s_smooth)
        background = np.percentile(smooth[valid], 10) + 1.28 * noise_smooth
        bright = np.percentile(smooth[valid], 99)
        threshold = background + max(self.k_fg * noise_smooth, self.rel_fg * (bright - background))
        foreground = ndi.binary_fill_holes(smooth > threshold)

        # Remove specks smaller than 15% of a full nucleus. Not larger: a plane can cut a
        # nucleus near its top or bottom, so small cross-sections are real.
        min_area = max(4, int(0.15 * np.pi * R**2))
        objects, n = ndi.label(foreground)
        keep = np.bincount(objects.ravel()) >= min_area
        keep[0] = False
        foreground = keep[objects]
        if not foreground.any():
            return np.zeros(plane.shape, np.uint16), no_seeds
        objects, n = ndi.label(foreground)

        # 2. Seeds: one starting point per nucleus, for the watershed.
        #    The LoG blob detector, tuned to radius R, peaks at nucleus centres. A seed is a
        #    local maximum within ~0.7 R that is k_seed noise sigmas above zero.
        s_log = R / np.sqrt(2)
        blob = -ndi.gaussian_laplace(plane, s_log) / (sigma_n * self._norm("log", s_log))
        spacing = 2 * max(1, int(round(0.7 * R))) + 1
        # (Same footprint as skimage's peak_local_max, without its slow Python spacing step.)
        seeds = np.argwhere((blob == ndi.maximum_filter(blob, size=spacing)) & (blob > self.k_seed) & foreground)

        # Every foreground object gets at least one seed (its brightest pixel), otherwise
        # the watershed would leave it out.
        seeded = np.zeros(n + 1, bool)
        seeded[objects[tuple(seeds.T)]] = True
        missing = np.flatnonzero(~seeded[1:]) + 1
        if missing.size:
            top = self._label_reduce(smooth, objects, n, np.maximum)
            hit = (objects > 0) & (smooth == top[objects]) & np.isin(objects, missing)
            _, first = np.unique(objects[hit], return_index=True)
            extra = np.argwhere(hit)[first]
            seeds = np.vstack([seeds, extra])

        # 3. Watershed: grow every seed into its own region, following the intensity
        #    'downhill' from the bright centre, until regions meet. That splits touching nuclei.
        markers = np.zeros(plane.shape, np.int32)
        markers[tuple(seeds.T)] = np.arange(1, len(seeds) + 1)
        labels = segmentation.watershed(-smooth, markers, mask=foreground,
                                        compactness=self.compactness)

        # 4. Clean up each object (edges, merged nuclei, haze).
        labels = self.refine_objects(labels, smooth, background, min_area, R)

        # 5. Drop objects that are not clearly above the camera noise. blob_noise comes from
        #    the difference with the neighbouring plane (see blob_noise), so structure such as
        #    haze does not inflate it. This is what keeps empty planes empty.
        if blob_noise and self.k_object > 0 and labels.max():
            raw_blob = blob * (sigma_n * self._norm("log", s_log))
            peak = self._label_reduce(raw_blob, labels, int(labels.max()), np.maximum)
            keep = peak >= self.k_object * blob_noise
            keep[0] = False
            labels = segmentation.relabel_sequential(np.where(keep[labels], labels, 0))[0].astype(np.uint16)
        return labels, seeds

    def refine_objects(self, labels, smooth, background, min_area, R):
        """Clean up the watershed objects of one plane.

        a. Edge: blurring makes bright nuclei look too big and dim ones too small with a
           single threshold. Cutting every object at half of its own height (above its
           local background) gives the true edge regardless of brightness or blur.
        b. Neck split: two touching nuclei can end up as one object shaped like a figure 8.
           Its distance map (distance to the edge) then has two peaks with a dip between
           them; we split at the dip.
        c. Flat-object test: diffuse haze (cytoplasm, out-of-focus light) can pass the
           threshold but barely stands out from its own surroundings. Nuclei do.

        a and c are done for all objects at once; only objects that fail a quick 'is this a
        single disk?' check go through the slower per-object split in b.
        """
        n = int(labels.max())
        if n == 0:
            return labels.astype(np.uint16)
        reach = max(1.0, R / 2)          # width of the ring around an object used as its local background
        ids = np.arange(1, n + 1)

        # a. Edge at edge_frac of each object's height above its local background
        #    (the darkest value within `reach` of the object).
        peak = self._label_reduce(smooth, labels, n, np.maximum)
        local_bg = np.maximum(background, self._label_reduce(smooth, segmentation.expand_labels(labels, reach), n, np.minimum))
        level = np.nan_to_num(local_bg + self.edge_frac * (peak - local_bg), nan=np.inf)
        mask = np.where(smooth >= level[labels], labels, 0)
        area = np.bincount(mask.ravel(), minlength=n + 1)
        mask[(area < min_area)[mask]] = 0

        # Quick disk test: in a single disk the deepest point (largest distance to the edge)
        # is close to the radius of a circle with the same area. Two touching disks reach at
        # most 0.71 of that, so anything above disk_like (0.8) cannot have a neck.
        inside = (mask > 0) & ~segmentation.find_boundaries(mask, mode="inner")
        depth = np.nan_to_num(self._label_reduce(ndi.distance_transform_edt(inside), mask, n, np.maximum), nan=0) + 1
        needs_split = (area >= min_area) & (depth < self.disk_like * np.sqrt(area / np.pi))
        needs_split[0] = False

        # b. Neck split, per object, only where needed. h_maxima keeps only peaks of the
        #    distance map that rise at least `h` above the dip between them, so a slightly
        #    irregular single nucleus is not cut. (The map is in 0.25 px steps, hence * 4.)
        h = int(round(max(1.0, self.neck_depth * R) * 4))
        next_id = n + 1
        H, W = mask.shape
        slices = ndi.find_objects(mask)
        for i in np.flatnonzero(needs_split):
            sl = slices[i - 1]
            if sl is None:
                continue
            # Work in the object's bounding box (+2 px margin): much cheaper than the whole plane.
            box = (slice(max(sl[0].start - 2, 0), min(sl[0].stop + 2, H)), slice(max(sl[1].start - 2, 0), min(sl[1].stop + 2, W)))
            obj = mask[box] == i
            distance = ndi.gaussian_filter(ndi.distance_transform_edt(ndi.binary_erosion(obj)).astype(np.float32), 1)
            parts, k = ndi.label(morphology.h_maxima(np.round(distance * 4).astype(np.uint16), h))
            if k < 2:
                continue
            parts = segmentation.watershed(-distance, parts, mask=obj)
            for j in range(2, k + 1):  # part 1 keeps label i, the others get new labels
                mask[box][parts == j] = next_id
                next_id += 1
        mask = segmentation.relabel_sequential(mask)[0]
        n = int(mask.max())
        if n == 0:
            return mask.astype(np.uint16)
        ids = np.arange(1, n + 1)
        area = np.bincount(mask.ravel(), minlength=n + 1)

        # c. Flat-object test. rel = (object - ring around it) / (object - plane background)
        #    is the share of an object's signal that is local. Nuclei score ~0.4-0.6, haze that
        #    sits on a broad 'pedestal' of light ~0.1-0.2. It is a ratio, so it does not depend
        #    on how bright or noisy the plane is.
        ring = segmentation.expand_labels(mask, reach)
        ring[mask > 0] = 0
        with np.errstate(invalid="ignore"):
            inside_mean = np.r_[0, ndi.mean(smooth, mask, ids)]
            around = np.r_[0, ndi.mean(smooth, ring, ids)]
        around = np.where(np.isfinite(around), around, background)  # fully enclosed: no evidence against it
        keep = (area >= min_area) & ((inside_mean - around) > self.min_rel_contrast * np.maximum(inside_mean - background, 1e-6))
        keep[0] = False
        return segmentation.relabel_sequential(np.where(keep[mask], mask, 0))[0].astype(np.uint16)

    # =====================================================================================
    # The whole stack
    # =====================================================================================

    def segment_stack(self, image, progress=True):
        """Segment every Z plane of a (Z, Y, X) image; returns (Z, Y, X) uint16 labels.

        Label numbers restart in every plane (2D labels); NucLogic's stitching links them
        into 3D nuclei. Diagnostics are stored in self.last. Set progress=False when the
        caller already shows a progress bar (alive_progress cannot nest bars).
        """
        stack = image.astype(np.float32)

        # Light blur along Z: each plane borrows signal from its neighbours (nuclei span
        # several planes, noise does not). Done in row blocks so the threads can share it.
        if self.z_sigma > 0:
            rows = np.linspace(0, stack.shape[1], self.workers + 1).astype(int)
            blurred = self._map(lambda k: ndi.gaussian_filter1d(stack[:, rows[k]:rows[k + 1]], self.z_sigma, axis=0),
                                range(self.workers))
            stack = np.concatenate(blurred, axis=1)

        # Some datasets are background-subtracted and clipped at 0, leaving large areas of
        # exact zeros. Statistics are only taken over 'valid' (non-zero) pixels so those
        # areas do not distort them. Data without such clipping uses every pixel.
        valid = stack > 0
        if valid.mean() > 0.99:
            valid[:] = True

        # Noise and quality of every plane; the best planes are used for calibration.
        noise = np.array(self._map(lambda z: self.noise_sigma(stack[z], valid[z]), range(len(stack))))
        quality = np.array(self._map(lambda z: self.quality(stack[z], noise[z], valid[z]), range(len(stack))))
        best = np.argsort(quality)[::-1][: self.top_k]

        if self.diameter:
            radii, history = np.empty(0), []
            self.radius = self.diameter / 2
        else:
            radii, history = self.calibrate_radius(stack[best], noise[best], valid[best])

        # Camera noise per plane for the final object test (needs R, so after calibration).
        # Uses the raw image: the Z blur would make neighbouring planes share noise.
        blob_noise = self.blob_noise(image, valid)
        labels = np.zeros(stack.shape, np.uint16)

        bar_context = alive_bar(len(stack), title="Segmenting planes") if progress else nullcontext(lambda: None)
        with bar_context as bar:
            def one(z):
                if quality[z] >= self.quality_floor:
                    labels[z], s = self.segment_plane(stack[z], noise[z], valid[z], blob_noise[z])
                else:
                    s = np.empty((0, 2), int)  # too little signal: leave the plane empty
                bar()
                return s

            seeds = self._map(one, range(len(stack)))

        self.last = dict(blob_noise=blob_noise, noise=noise, quality=quality, calibration_planes=best,
                         radii=radii, radius=self.radius, radius_history=history, seeds=seeds)
        return labels
