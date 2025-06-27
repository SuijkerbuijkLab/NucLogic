# Function to calculate phenotypes of all cells based on image channels


def phenotype(data, k_channel1=0, k_channel2=0.5):

    phenotype = []
    count_phenotype_c1 = 0
    count_phenotype_c2 = 0
    count_phenotype_c1and2 = 0
    count_phenotype_empty = 0

    for _, row in data.iterrows():
        VolTot = row["volume"]
        VolCh1 = row["area_WT"]
        avgCh1 = row["raw_WT"] / VolCh1 if VolCh1 > 0 else 0
        VolCh2 = row["area_CRC"]
        avgCh2 = row["raw_CRC"] / VolCh2 if VolCh2 > 0 else 0

        pheno = "-"
        qa = 0
        qb = 0

        if VolCh1 > k_channel1 * VolTot:
            # Fenotipo WT
            pheno_c1 = "WT"
            count_phenotype_c1 += 1
            qa = 1
        else:
            pheno_c1 = "-"

        if VolCh2 > k_channel2 * VolTot:
            # Fenotipo CRC
            pheno_c2 = "CRC"
            pheno = "CRC"
            count_phenotype_c2 += 1
            qb = 1
        else:
            pheno_c2 = "-"

        if qa + qb == 2:
            if avgCh1 >= 2 * avgCh2:
                pheno = "WT"
            else:
                pheno = "CRC"
            count_phenotype_c1and2 += 1
        elif pheno == "-":
            count_phenotype_empty += 1

        if qa == 1 and qb == 0:
            pheno = "WT"

        phenotype.append(pheno)

    data["Phenotype"] = phenotype

    return (
        data,
        count_phenotype_c1,
        count_phenotype_c2,
        count_phenotype_c1and2,
        count_phenotype_empty,
    )
