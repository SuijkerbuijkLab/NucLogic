run("Collect Garbage");
run("CLIJ2 Macro Extensions", "cl_device=");
Ext.CLIJ2_clear();

// Define variables
base = "base"; 
proj = "proj"; 
projXZ = "projXZ"; 
temp = "temp"; 
blur = "blur";

// Set measurement options
run("Set Measurements...", "area mean standard modal min centroid center perimeter bounding fit shape feret's integrated median skewness kurtosis area_fraction stack limit redirect=None decimal=2");

// Get image properties
getDimensions(width, height, channels, slices, frames);
nameRaw = File.nameWithoutExtension; name = "name"; 
rename(name);
getVoxelSize(dxr, dyr, dzr, unitr);
dtr = Stack.getFrameInterval(); 
Stack.getUnits(X, Y, Z, TimeU, Value); unitTime = TimeU; 

// Handle bit depth and set dynamic range
minDynRange = 1; bitAr = bitDepth();
if (bitAr == 8) {
    maxDynRange = 255;
} else if (bitAr <= 16) {
    maxDynRange = 65536;
} else {
    print("Only 8-bit and 16-bit images are supported. Converting to 8-bit.");
    run("8-bit");
    maxDynRange = 255;
}

// Clear results and logs
run("Clear Results");
if (isOpen("Log")) {
    selectWindow("Log");
    run("Close");
}

// Filtering options dialog
Dialog.createNonBlocking("The macro will start after clik ok. Inspect channel info and decide if you want to apply an automated processing of them. Remember that the image will be saved in the folder where the input is stored. Buena suerte ;)");
Dialog.addMessage("Check the reference channel (e.g., nuclei). Others are optional.", 12, "purple");
Dialog.addString("OutpuFolder", "frames_extracted");
Dialog.addNumber("Ref. channel", 3);
Dialog.addNumber("WT channel", 2);
Dialog.addNumber("CRC channel", 1);
Dialog.addMessage("It is recommended to apply it later using the macroIC", 12, "purple");
Dialog.addCheckbox("Apply filtering", false);
Dialog.addNumber("Kernel radius (not size)", 3);
Dialog.addMessage("Projections are generated without no impact on processing speed. A separate program handles this task post-processing (longer runtime).", 8, "purple");
Dialog.addCheckbox("Do you want to create the prejction for later selections/crops?", true);
Dialog.show();
outputFolder = Dialog.getString();
ch_ref = Dialog.getNumber();
wt_ref = Dialog.getNumber();
c_ref = Dialog.getNumber();
doFiltering = Dialog.getCheckbox();
minR = Dialog.getNumber();
doProjection = Dialog.getCheckbox();
//
setBatchMode("hide");
//
pathBase = getDir("image");
if ((pathBase == "C:\\") | (pathBase == "")) {
    pathBase = getDirectory("Output folder not specified. Choose one directory"));
}
pathO = pathBase+ "/frames-" + outputFolder; 
//pathO = "D:/ImageProcessing_codes/TL_impo/fullOrganoids/fullPipeline/prueba/proc0_new"; 
if (!File.isDirectory(pathO)) {
    File.makeDirectory(pathO);
}
//
// Main processing loop
for (t = 0; t < frames; t++) {
    selectImage(name);
    Stack.setFrame(t + 1);
    mergeString = "";
    for (c = 1; c <= channels; c++) {
        selectImage(name);
        Stack.setChannel(c);
        Ext.CLIJ2_pushCurrentZStack(name);
        Ext.CLIJ2_pull(name);
        rename(base);
        //selectImage(name); run("Duplicate...", "title="+base+" duplicate channels="+ch_ref+" frames="+(t+1)+"");  
        //selectImage(name); run("Duplicate...", "title="+base+" duplicate channels="+c+" frames="+(t+1)+"");  
        // Apply filtering if enabled
        if (doFiltering) {
			selectImage(base);
			newS = slices;
			for (z = 1; z <= newS; z++) {    
				showProgress(z, newS);
				run("Select None"); 
				selectImage(base); setSlice(z); 
				selectImage(base); run("Duplicate...", "title=" + temp + " duplicate range=" + z + "-" + z + "");       
				//
				selectImage(temp); run("Median...", "radius=" + minR + "");
				selectImage(temp); run("Enhance Contrast...", "saturated=0.1 normalize");            
				selectImage(temp); setThreshold(minDynRange, maxDynRange);    
				List.setMeasurements("limit");
				valT = getValue("Mean limit");
				valT = round(valT); 
				selectImage(temp); resetThreshold();
				selectImage(temp); run("Subtract...", "value=" + valT + ""); 
				selectImage(temp); rename(temp + "-" + z); 
			}
			run("Images to Stack", "name=" + blur + " title=" + temp + "-");     
			selectImage(base); close(); 
			selectImage(blur); rename(base); 				
		} 
        // Save processed channel
		if (c == ch_ref){
			cInf = "ref"; 
		}else if (c == wt_ref){
			cInf = "WT"; 
		}else if (c == c_ref){
			cInf = "CRC"; 
		}else{
			cInf = c; 
		}
		nameFileR="Channel-" + cInf + "-frame-" + t; 
		selectImage(base); saveAs("Tiff", pathO + "/" + nameFileR); 	
		selectImage(nameFileR + ".tif"); close(); 		
		// 
        Ext.CLIJ2_clear();
    }
}
selectImage(name); close(); 
print("The macro has been processed and the frames and channels saved in " + pathO);

widthR = width; 
heightR = height; 
slicesR = slices; 
channelsR = channels; 
framesR = frames; 

run("Collect Garbage");
run("CLIJ2 Macro Extensions", "cl_device=");
Ext.CLIJ2_clear();

// Create projections if enabled
if (doProjection) {
    for (t = 0; t < framesR; t++) {
    	//
    	// open the image 
        file_input_frame = pathO + "/Channel-ref-frame-" + t + ".tif"; 
        run("TIFF Virtual Stack...", "open=["+file_input_frame+"]");
        getDimensions(width, height, channels, slices, frames); 
        name1=getInfo("image.title");  name=name1; if (name1=="Composite"){ name="CompositeC";}
        rename(""+name+""); nameFileR=""+name1+"";	nameIm= substring(name, 0, lengthOf(name) - 4);
        selectImage(name); bitA=bitDepth(); minDynRange=1;
        if (bitA==8){	maxDynRange=255; run("8-bit");
        }else if (bitA<=16){maxDynRange=65536; 
        }else{run("8-bit"); maxDynRange=255;}
        selectImage(name); getVoxelSize(dx, dy, dz, unit); selectImage(name); dT=Stack.getFrameInterval(); 
        if (is("global scale") == 0){selectImage(name);run("Set Scale...", "distance=0 known=0 pixel=1 unit=pixel");}; 	
        //
        Ext.CLIJ2_push(name); selectImage(name); close(); 
        Ext.CLIJ2_maximumZProjection(name, proj);
        Ext.CLIJ2_pull(proj); Ext.CLIJ2_release(proj);
        rename(proj + "-" + (t + 1)); 
        //
        Ext.CLIJ2_resliceTop(name, temp);
        Ext.CLIJ2_maximumZProjection(temp, projXZ); Ext.CLIJ2_release(temp);
        Ext.CLIJ2_pull(projXZ); Ext.CLIJ2_release(projXZ);
        rename(projXZ + "-" + (t + 1));
        //
        Ext.CLIJ2_release(name);             
        //
    }
    // Save projections
    run("Images to Stack", "name=projection-out title=" + proj + "- fill=white bicubic");
    run("Images to Stack", "name=projectionXZ-out title=" + projXZ + "- fill=white bicubic");
    //
    selectImage("projection-out"); 
    run("Label...", "format=0000 starting=0 interval=1 x="+round(widthR * 0.05)+" y="+round(heightR * 0.1)+" font="+round(minOf(widthR, heightR) * 0.02)+" text=Frame range=1-"+framesR+"");
    selectImage("projectionXZ-out");  
    run("Label...", "format=0000 starting=0 interval=1 x="+round(widthR * 0.05)+" y="+round(heightR * 0.1)+" font="+round(minOf(widthR, heightR) * 0.02)+" text=Frame range=1-"+framesR+"");
	//
	selectImage("projection-out"); saveAs("Tiff", pathO + "/" + proj+"XY"); 	
	selectImage(proj+"XY.tif"); close(); 
	selectImage("projectionXZ-out"); saveAs("Tiff", pathO + "/" + projXZ); 	
	selectImage(projXZ+".tif"); close(); 
}
setBatchMode("exit and display");
//
// Construir texto con los datos
info = "Nombre: " + name + "\n" +
       "Dimensiones:\n" +
       "  Width (X): " + widthR + "\n" +
       "  Height (Y): " + heightR + "\n" +
       "  Z-Stack (Profundidad): " + slicesR + "\n" +
       "  Canales: " + channelsR + "\n" +
       "  Tiempos: " + framesR + "\n" +
       "  BitsPerPixel: " + bitAr + "\n" + 
       "  dx: " + dxr + " " + unitr + "\n" +
       "  dy: " + dyr + " " + unitr + "\n" +
       "  dz: " + dzr + " " + unitr + "\n" +
       "  unit: " + unitr + "\n" +
       "  dt: " + dtr + " " + unitTime + "\n" +
       "  dt_unit: " + unitTime + "\n" +  // Aquí se agregó el "+" que faltaba
       "Parameters:\n" +
       "  doFiltering: " + doFiltering + "\n" +
       "  kernel radius (median filter): " + doProjection + "\n" +
       "  Channel of reference (nuclei): " + ch_ref + "\n" +
       "  Channel of Phenotype 1 (CRC): " + c_ref + "\n" +
       "  Channel of Phenotype 2 (WT): " + wt_ref + "\n" +
       "  projections?: " + doProjection + "\n";


// Ruta donde se guardará el archivo .txt
pathOm = pathO + "/archivo_metadata.txt";
File.saveString(info, pathOm);
// Cleanup
setBatchMode("exit and display");
Ext.CLIJ2_clear();
run("Clear Results");
if (isOpen("Results")) {
    selectWindow("Results");
    run("Close");
}
