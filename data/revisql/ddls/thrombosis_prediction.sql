CREATE TABLE Laboratory (
    `ID` integer, -- identification of the patient.
    `Date` date, -- Date of the laboratory tests (YYMMDD).
    `GOT` integer, -- AST glutamic oxaloacetic transaminase. Commonsense evidence:. Normal range: N < 60.
    `GPT` integer, -- ALT glutamic pyruvic transaminase. Commonsense evidence:. Normal range: N < 60.
    `LDH` integer, -- lactate dehydrogenase. Commonsense evidence:. Normal range: N < 500.
    `ALP` integer, -- alkaliphophatase. Commonsense evidence:. Normal range: N < 300.
    `TP` real, -- total protein. Commonsense evidence:. Normal range: 6.0 < N < 8.5.
    `ALB` real, -- albumin. Commonsense evidence:. Normal range: 3.5 < N < 5.5.
    `UA` real, -- uric acid. Commonsense evidence:. Normal range: N > 8.0 (Male)N > 6.5 (Female).
    `UN` integer, -- urea nitrogen. Commonsense evidence:. Normal range: N < 30.
    `CRE` real, -- creatinine. Commonsense evidence:. Normal range: N < 1.5.
    `T-BIL` real, -- total bilirubin. Commonsense evidence:. Normal range: N < 2.0.
    `T-CHO` integer, -- total cholesterol. Commonsense evidence:. Normal range: N < 250.
    `TG` integer, -- triglyceride. Commonsense evidence:. Normal range: N < 200.
    `CPK` integer, -- creatinine phosphokinase. Commonsense evidence:. Normal range: N < 250.
    `GLU` integer, -- blood glucose. Commonsense evidence:. Normal range: N < 180.
    `WBC` real, -- White blood cell. Commonsense evidence:. Normal range: 3.5 < N < 9.0.
    `RBC` real, -- Red blood cell. Commonsense evidence:. Normal range: 3.5 < N < 6.0.
    `HGB` real, -- Hemoglobin. Commonsense evidence:. Normal range: 10 < N < 17.
    `HCT` real, -- Hematoclit. Commonsense evidence:. Normal range: 29 < N < 52.
    `PLT` integer, -- platelet. Commonsense evidence:. Normal range: 100 < N < 400.
    `PT` real, -- prothrombin time. Commonsense evidence:. Normal range: N < 14.
    `APTT` integer, -- activated partial prothrombin time. Commonsense evidence:. Normal range: N < 45.
    `FG` real, -- fibrinogen. Commonsense evidence:. Normal range: 150 < N < 450.
    `PIC` integer, -- No description.
    `TAT` integer, -- No description.
    `TAT2` integer, -- No description.
    `U-PRO` text, -- proteinuria. Commonsense evidence:. Normal range: 0 < N < 30.
    `IGG` integer, -- Ig G. Commonsense evidence:. Normal range: 900 < N < 2000.
    `IGA` integer, -- Ig A. Commonsense evidence:. Normal range: 80 < N < 500.
    `IGM` integer, -- Ig M. Commonsense evidence:. Normal range: 40 < N < 400.
    `CRP` text, -- C-reactive protein. Commonsense evidence:. Normal range: N= -, +-, or N < 1.0.
    `RA` text, -- Rhuematoid Factor. Commonsense evidence:. Normal range: N= -, +-.
    `RF` text, -- RAHA. Commonsense evidence:. Normal range: N < 20.
    `C3` integer, -- complement 3. Commonsense evidence:. Normal range: N > 35.
    `C4` integer, -- complement 4. Commonsense evidence:. Normal range: N > 10.
    `RNP` text, -- anti-ribonuclear protein. Commonsense evidence:. Normal range: N= -, +-.
    `SM` text, -- anti-SM. Commonsense evidence:. Normal range: N= -, +-.
    `SC170` text, -- anti-scl70. Commonsense evidence:. Normal range: N= -, +-.
    `SSA` text, -- anti-SSA. Commonsense evidence:. Normal range: N= -, +-.
    `SSB` text, -- anti-SSB. Commonsense evidence:. Normal range: N= -, +-.
    `CENTROMEA` text, -- anti-centromere. Commonsense evidence:. Normal range: N= -, +-.
    `DNA` text, -- anti-DNA. Commonsense evidence:. Normal range: N < 8.
    `DNA-II` integer, -- anti-DNA. Commonsense evidence:. Normal range: N < 8.
    primary key (ID, Date),
    foreign key (ID) references Patient (ID)
);

CREATE TABLE Patient (
    `ID` integer, -- identification of the patient.
    `SEX` text, -- Sex. F: female; M: male.
    `Birthday` date, -- Birthday.
    `Description` date, -- the first date when a patient data was recorded. null or empty: not recorded.
    `First Date` date, -- the date when a patient came to the hospital.
    `Admission` text, -- patient was admitted to the hospital (+) or followed at the outpatient clinic (-).
    `Diagnosis` text, -- disease names.
    primary key (ID)
);

CREATE TABLE Examination (
    `ID` integer, -- identification of the patient.
    `Examination Date` date , -- Examination Date.
    `aCL IgG` real, -- anti-Cardiolipin antibody (IgG). anti-Cardiolipin antibody (IgG) concentration.
    `aCL IgM` real, -- anti-Cardiolipin antibody (IgM). anti-Cardiolipin antibody (IgM) concentration.
    `ANA` integer, -- anti-nucleus antibody. anti-nucleus antibody concentration.
    `ANA Pattern` text, -- pattern observed in the sheet of ANA examination.
    `aCL IgA` integer, -- anti-Cardiolipin antibody (IgA) concentration.
    `Diagnosis` text, -- disease names.
    `KCT` text, -- measure of degree of coagulation. +: positive. -: negative.
    `RVVT` text, -- measure of degree of coagulation. +: positive. -: negative.
    `LAC` text, -- measure of degree of coagulation. +: positive. -: negative.
    `Symptoms` text, -- other symptoms observed.
    `Thrombosis` integer, -- degree of thrombosis. 0: negative (no thrombosis). 1: positive (the most serious). 2: positive (severe)3: positive (mild).
    foreign key (ID) references Patient (ID)
);

