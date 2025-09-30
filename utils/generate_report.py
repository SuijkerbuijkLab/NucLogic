# Function that makes some plots and saves them in a report pdf

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def generate_report(df, time_interval=1, output_path="report.pdf"):
    times = list(df.index.values) * time_interval

    with PdfPages(output_path) as pdf:
        # Plot % of cells over time
        plt.figure()
        plt.plot(times, df["%wt"], "m-", marker="o", label="WT")
        plt.plot(times, df["%crc"], "g-", marker="o", label="CRC")
        plt.title("% of cells per organoid over time")
        plt.ylabel("% cells per organoid")
        plt.xlabel("Time (h)")
        plt.legend()
        pdf.savefig()  # Save the current figure
        plt.close()

        # WT relative growth
        plt.figure()
        plt.plot(times, df["relative_wt"], "m-", marker="o", label="WT")
        plt.title("WT relative cell growth")
        plt.ylabel("Number of cells per organoid\n(normalized to t=0)")
        plt.xlabel("Time (h)")
        plt.legend()
        pdf.savefig()
        plt.close()

        # CRC relative growth
        plt.figure()
        plt.plot(times, df["relative_crc"], "g-", marker="o", label="CRC")
        plt.title("CRC relative cell growth")
        plt.ylabel("Number of cells per organoid\n(normalized to t=0)")
        plt.xlabel("Time (h)")
        plt.legend()
        pdf.savefig()
        plt.close()
