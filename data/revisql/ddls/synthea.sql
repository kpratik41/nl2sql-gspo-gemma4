CREATE TABLE procedures (
    `DATE` date, -- the date of the procedure. yyyy-mm-dd.
    `PATIENT` text, -- the patient id.
    `ENCOUNTER` text, -- the medical encounter id.
    `CODE` integer, -- the code of the procedure.
    `DESCRIPTION` text, -- the description of the procedure.
    `REASONCODE` integer, -- reason code. the code of the reason.
    `REASONDESCRIPTION` text, -- reason description. the description of the reason why the patient take the procedure.
    foreign key (ENCOUNTER) references encounters(ID),
    foreign key (PATIENT) references patients(patient)
);

CREATE TABLE immunizations (
    `DATE` date, -- the date of the immunization. yyyy-mm-dd.
    `PATIENT` text, -- the patient id.
    `ENCOUNTER` text, -- the medical encounter id.
    `CODE` integer, -- the code of the immunization.
    `DESCRIPTION` text, -- the description of the immunization.
    primary key (DATE, PATIENT, ENCOUNTER, CODE),
    foreign key (ENCOUNTER) references encounters(ID),
    foreign key (PATIENT) references patients(patient)
);

CREATE TABLE conditions (
    `START` date, -- the start date of the allergy. mm/dd/yy.
    `STOP` date, -- the stop date of the allergy. mm/dd/yy.
    `PATIENT` text, -- the patient id.
    `ENCOUNTER` text, -- the medical encounter id.
    `CODE` integer, -- the code of the condition.
    `DESCRIPTION` text, -- the description of the patient condition.
    foreign key (ENCOUNTER) references encounters(ID),
    foreign key (PATIENT) references patients(patient),
    foreign key (DESCRIPTION) references all_prevalences(ITEM)
);

CREATE TABLE all_prevalences (
    `ITEM` text, -- the prevalent disease.
    `POPULATION TYPE` text, -- population type. the population type - LIVING.
    `OCCURRENCES` integer, -- the number of occurrences.
    `POPULATION COUNT` integer, -- population count. the number of the counted populations.
    `PREVALENCE RATE` real, -- prevalence rate. the prevalence rate. prevalence rate = occurrences / population_count.
    `PREVALENCE PERCENTAGE` real, -- prevalence percentage. the prevalence percentage. prevalence rate = occurrences / population_count * 100.
    primary key (ITEM)
);

CREATE TABLE encounters (
    `ID` text, -- the unique id of the encounter.
    `DATE` date, -- the date of the encounter. yyyy-mm-dd.
    `PATIENT` text, -- the patient id.
    `CODE` integer, -- the code of the care plan.
    `DESCRIPTION` text, -- the description of the care plan.
    `REASONCODE` integer, -- the reason code.
    `REASONDESCRIPTION` text, -- reason description. the description of the reason why the patient needs the care plan.
    primary key (ID),
    foreign key (PATIENT) references patients(patient)
);

CREATE TABLE careplans (
    `ID` text, -- the unique id of the care plan.
    `START` date, -- the start date of the care plan. yyyy-mm-dd.
    `STOP` date, -- the stop date of the care plan. yyyy-mm-dd. care plan period:. stop - start.
    `PATIENT` text, -- the patient id.
    `ENCOUNTER` text, -- the medical encounter id.
    `CODE` real, -- the code of the care plan.
    `DESCRIPTION` text, -- the description of the care plan.
    `REASONCODE` integer, -- the reason code.
    `REASONDESCRIPTION` text, -- reason description. the description of the reason why the patient needs the care plan.
    foreign key (ENCOUNTER) references encounters(ID),
    foreign key (PATIENT) references patients(patient)
);

CREATE TABLE medications (
    `START` date, -- the start date of the care plan. yyyy-mm-dd.
    `STOP` date, -- the stop date of the care plan. yyyy-mm-dd. Time of taking medicine.
    `PATIENT` text, -- the patient id.
    `ENCOUNTER` text, -- the medical encounter id.
    `CODE` integer, -- the code of the medication.
    `DESCRIPTION` text, -- the description of the medication.
    `REASONCODE` integer, -- the reason code.
    `REASONDESCRIPTION` text, -- reason description. the description of the reason why the patient take the medication.
    primary key (START, PATIENT, ENCOUNTER, CODE),
    foreign key (ENCOUNTER) references encounters(ID),
    foreign key (PATIENT) references patients(patient)
);

CREATE TABLE patients (
    `patient` text, -- the unique id for the patient.
    `birthdate` date, -- birth date. the birth date of the patient.
    `deathdate` date, -- death date. the death date of the patient.  the age of the patient = death year - birth year.  if null, it means this patient is still alive.
    `ssn` text, -- social security number. the social security number of the patient.
    `drivers` text, -- the driver number of the patient. if not, this patient doesn't have driving license.
    `passport` text, -- the passport number. if not, this patient cannot go abroad, vice versa.
    `prefix` text, -- the prefix.
    `first` text, -- the first name.
    `last` text, -- the last name. full name = first + last.
    `suffix` text, -- the suffix of the patient. if suffix = PhD, JD, MD, it means this patient has doctoral degree. Otherwise, this patient is not.
    `maiden` text, -- the maiden name of the patient. Only married women have the maiden name.
    `marital` text, -- the marital status of the patient.  M: married.  S: single.
    `race` text, -- the race of the patient.
    `ethnicity` text, -- the ethnicity of the patient.
    `gender` text, -- the gender of the patient.
    `birthplace` text, -- birth place. the birth place.
    `address` text, -- the specific address.
    primary key (patient)
);

CREATE TABLE allergies (
    `START` text, -- the start date of the allergy. mm/dd/yy.
    `STOP` text, -- the stop date of the allergy. mm/dd/yy. allergy period = stop - start.
    `PATIENT` text, -- the patient id.
    `ENCOUNTER` text, -- the medical encounter id.
    `CODE` integer, -- the code of the allergy.
    `DESCRIPTION` text, -- the description of the allergy.
    primary key (PATIENT, ENCOUNTER, CODE),
    foreign key (ENCOUNTER) references encounters(ID),
    foreign key (PATIENT) references patients(patient)
);

CREATE TABLE claims (
    `ID` text, -- the unique id of the claim.
    `PATIENT` text, -- the patient id.
    `BILLABLEPERIOD` date, -- billable period. the start date of the billable. yyyy-mm-dd.
    `ORGANIZATION` text, -- the claim organization.
    `ENCOUNTER` text, -- the medical encounter id.
    `DIAGNOSIS` text, -- the diagnosis.
    `TOTAL` integer, -- the length of the billable period.
    primary key (ID)
);

CREATE TABLE observations (
    `DATE` date, -- the date of the observation. yyyy-mm-dd.
    `PATIENT` text, -- the patient id.
    `ENCOUNTER` text, -- the medical encounter id.
    `CODE` text, -- the code of the observation type.
    `DESCRIPTION` text, -- the description of the observation.
    `VALUE` real, -- the observation value.
    `UNITS` text, -- the units of the observation value. DESCRIPTION + VALUE + UNITS could be a fact:. e.g.: body height of patient xxx is 166.03 cm:. body height is in DESCRIPTION. 166.03 is in VALUE. cm is in UNITS.
    foreign key (ENCOUNTER) references encounters(ID),
    foreign key (PATIENT) references patients(patient)
);

