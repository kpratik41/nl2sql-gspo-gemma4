CREATE TABLE Interactions (
    `GeneID1` text, -- gene id 1. identifier number of genes.
    `GeneID2` text, -- gene id 2. identifier number of genes.
    `Type` text, -- Type.
    `Expression_Corr` real, -- Expression correlation score. range: (0,1). if the value is the positive then it's "positively correlated". if the value is the negative then it's "negatively correlated". if the value is very high positively, it means two genes are highly correlated.
    primary key (GeneID1, GeneID2),
    foreign key (GeneID1) references Classification (GeneID)
    foreign key (GeneID2) references Classification (GeneID)
);

CREATE TABLE Classification (
    `GeneID` text, -- gene id. unique identifier number of genes.
    `Localization` text, -- location.
    primary key (GeneID)
);

CREATE TABLE Genes (
    `GeneID` text, -- gene id. identifier number of genes.
    `Essential` text, -- essential.
    `Class` text, -- class.
    `Complex` text, -- Complex.
    `Phenotype` text, -- Phenotype. ?: means: doesn't exit the phenotype.
    `Motif` text, -- Motif.
    `Chromosome` text, -- Chromosome.
    `Function` text, -- Function.
    `Localization` text, -- No description.
    foreign key (GeneID) references Classification (GeneID)
);

