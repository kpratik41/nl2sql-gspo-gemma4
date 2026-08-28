CREATE TABLE state_sector_grads (
    `stateid` integer, -- state id. state FIPS (Federal Information Processing System) code.
    `state` text, -- state name.
    `state_abbr` text, -- No description.
    `control` text, -- Public,. Private not-for-profit,. Private for-profit.
    `level` text, -- Level of institution. 4-year: bachelor's degree. 2-year: associate's degree.
    `year` integer, -- year of data release.
    `gender` text, -- gender of students. B' = both genders; 'M' = male; 'F' = female.
    `race` text, -- race/ethnicity of students. 'X' = all students; 'Ai' = American Indian; 'A' = Asian; 'B' = Black; 'H' = Hispanic; 'W' = White.
    `cohort` text, -- degree-seeking cohort type.  '4y bach' = Bachelor's/equivalent-seeking cohort at 4-year institutions;  '4y other' = Students seeking another type of degree or certificate at a 4-year institution;  '2y all' = Degree-seeking students at 2-year institutions. 4-year degree is bachelor's degree. 2-year degree is associate's degree.
    `grad_cohort` text, -- graduation cohort. Number of first-time, full-time, degree-seeking students in the cohort being tracked, minus any exclusions.
    `grad_100` text, -- graduation 100. Number of students who graduated within 100 percent of normal/expected time.
    `grad_150` text, -- graduation 150. Number of students who graduated within 150 percent of normal/expected time.
    `grad_100_rate` text, -- Percentage of students who graduated within 100 percent of normal/expected time.
    `grad_150_rate` text, -- Percentage of students who graduated within 150 percent of normal/expected time.
    `grad_cohort_ct` integer, -- graduation cohort count. Number of institutions with data included in the cohort.
    foreign key (state) references institution_details(state),
    foreign key (stateid) references state_sector_details(stateid)
);

CREATE TABLE institution_details (
    `unitid` integer, -- Unit ID number. unique Education Unit ID number.
    `chronname` text, -- Institution name.
    `city` text, -- Institution city.
    `state` text, -- Institution state.
    `level` text, -- Level of institution. 4-year: bachelor's degree. 2-year: associate's degree.
    `control` text, -- Control of institution. Public,. Private not-for-profit,. Private for-profit.
    `basic` text, -- Carnegie Foundation for the Advancement of Teaching Basic Classification (2010 version).
    `hbcu` text, -- Historically Black College and Universities. Denotes Historically Black College and Universities.
    `flagship` text, -- Denotes Public flagship institutions.
    `long_x` real, -- Institution longitude.
    `lat_y` real, -- Institution latitude. institute Coordinate.
    `site` text, -- Institution Web site address.
    `student_count` integer, -- student count. Total number of undergraduates in 2010.
    `awards_per_value` real, -- awards per value. Completions per 100 FTE (full-time equivalency) undergraduate students (average 2011, 2012, and 2013).
    `awards_per_state_value` real, -- awards per state value. Completions per 100 FTE (full-time equivalency) undergraduate students, state and sector average.  if his institute's completion (or graduation) rate exceeds the average for the state and sector in which it belongs:. awards_per_value > awards_per_state_value.  if his institute's completion (or graduation) rate falls below the average for the state and sector in which it belongs:. awards_per_value < awards_per_state_value.
    `awards_per_natl_value` real, -- awards per national value. Completions per 100 FTE (full-time equivalency) undergraduate students, national sector average.  if his institute's completion (or graduation) rate exceeds the average for the nationalin which it belongs:. awards_per_value > awards_per_natl_value.  if his institute's completion (or graduation) rate falls below the average for the state and industry in which it belongs:. awards_per_value < awards_per_natl_value.
    `exp_award_value` integer, -- expected award value. Estimated educational spending per academic award in 2013. Includes all certificates and degrees. expenses related to instruction, research, public service, student services, academic support, institutional support, operations and maintenance. Includes all certificates and degrees.
    `exp_award_state_value` integer, -- expected award state value. Spending per completion, state and sector average.
    `exp_award_natl_value` integer, -- expected award national value. Spending per completion, national sector average.
    `exp_award_percentile` integer, -- No description.
    `ft_pct` real, -- Full-time percentage. Percentage of undergraduates who attend full-time.
    `fte_value` integer, -- Full-time percentage. total number of full-time equivalent undergraduates.
    `fte_percentile` integer, -- No description.
    `med_sat_value` text, -- median SAT value. Median estimated SAT value for incoming students.
    `med_sat_percentile` text, -- median SAT percentile. Institution's percent rank for median SAT value within sector.
    `aid_value` integer, -- aid value. The average amount of student aid going to undergraduate recipients.
    `aid_percentile` integer, -- aid percentile. Institution's percent rank for average amount of student aid going to undergraduate recipients within sector.
    `endow_value` text, -- endow value. End-of-year endowment value per full-time equivalent student.
    `endow_percentile` text, -- endow percentile. Institution's percent rank for endowment value per full-time equivalent student within sector.
    `grad_100_value` real, -- graduation 100 value. percentage of first-time, full-time, degree-seeking undergraduates who complete a degree or certificate program within 100 percent of expected time (bachelor's-seeking group at 4-year institutions). %,. lower means harder to graduate for bachelor.
    `grad_100_percentile` integer, -- graduation 100 percentile. Institution's percent rank for completers within 100 percent of normal time within sector.
    `grad_150_value` real, -- graduation 150 value. percentage of first-time, full-time, degree-seeking undergraduates who complete a degree or certificate program within 150 percent of expected time (bachelor's-seeking group at 4-year institutions). %,. lower means harder to graduate for bachelor.
    `grad_150_percentile` integer, -- graduation 150 percentile. Institution's percent rank for completers within 150 percent of normal time within sector.
    `pell_value` real, -- pell value. percentage of undergraduates receiving a Pell Grant.
    `pell_percentile` integer, -- pell percentile. Institution's percent rank for undergraduate Pell recipients within sector.
    `retain_value` real, -- retain value. share of freshman students retained for a second year.
    `retain_percentile` integer, -- retain percentile. Institution's percent rank for freshman retention percentage within sector.
    `ft_fac_value` real, -- full time faculty value. Percentage of employees devoted to instruction, research or public service who are full-time and do not work for an associated medical school.
    `ft_fac_percentile` integer, -- full time faculty percentile. Institution's percent rank for full-time faculty share within sector.
    `vsa_year` text, -- voluntary system of accountability year. Most recent year of Student Success and Progress Rate data available from the Voluntary System of Accountability.
    `vsa_grad_after4_first` text, -- voluntary system of accountability after 4 year first time. First-time, full-time students who graduated from this institution within four years.
    `vsa_grad_elsewhere_after4_first` text, -- voluntary system of accountability graduation elsewhere after 4 year first time. First-time, full-time students who graduated from another institution within four years.
    `vsa_enroll_after4_first` text, -- voluntary system of accountability enrollment after 4 year first time. First-time, full-time students who are still enrolled at this institution after four years.
    `vsa_enroll_elsewhere_after4_first` text, -- voluntary system of accountability enrollment elsewhere after 4 year first time. First-time, full-time students who are enrolled at another institution after four years.
    `vsa_grad_after6_first` text, -- voluntary system of accountability graduation elsewhere after 6 year first time. First-time, full-time students who graduated from this institution within six years. null: not available.
    `vsa_grad_elsewhere_after6_first` text, -- voluntary system of accountability graduation elsewhere after 6 year first time. First-time, full-time students who graduated from another institution within six years.
    `vsa_enroll_after6_first` text, -- voluntary system of accountability enrollment after 6 year first time. First-time, full-time students who are still enrolled at this institution after six years.
    `vsa_enroll_elsewhere_after6_first` text, -- voluntary system of accountability enrollment elsewhere after 6 year first time. First-time, full-time students who are enrolled at another institution after six years.
    `vsa_grad_after4_transfer` text, -- voluntary system of accountability transfer after 6 year first time. Full-time transfer students who graduated from this institution within four years.
    `vsa_grad_elsewhere_after4_transfer` text, -- voluntary system of accountability graduation elsewhere after 4 year. Full-time transfer students who graduated from this institution within four years.
    `vsa_enroll_after4_transfer` text, -- voluntary system of accountability enrollment after 4 years transfer. Full-time transfer students who are still enrolled at this institution after four years.
    `vsa_enroll_elsewhere_after4_transfer` text, -- voluntary system of accountability enrollment elsewhere after 4 years transfer. Full-time transfer students who are enrolled at another institution after four years.
    `vsa_grad_after6_transfer` text, -- voluntary system of accountability enrollment elsewhere after 6 years transfer. Full-time transfer students who graduated from this institution within six years.
    `vsa_grad_elsewhere_after6_transfer` text, -- voluntary system of accountability graduation elsewhere after 6 years transfer. Full-time transfer students who graduated from another institution within six years.
    `vsa_enroll_after6_transfer` text, -- voluntary system of accountability enrollment after 6 years transfer. Full-time transfer students who are still enrolled at this institution after six years.
    `vsa_enroll_elsewhere_after6_transfer` text, -- voluntary system of accountability enrollment elsewhere after 6 years transfer. Full-time transfer students who are enrolled at another institution after six years.
    `similar` text, -- No description.
    `state_sector_ct` integer, -- No description.
    `carnegie_ct` integer, -- No description.
    `counted_pct` text, -- No description.
    `nicknames` text, -- No description.
    `cohort_size` integer, -- No description.
    primary key (unitid)
);

CREATE TABLE state_sector_details (
    `stateid` integer, -- state id. state FIPS (Federal Information Processing System) code.
    `state` text, -- state name.
    `state_post` text, -- No description.
    `level` text, -- Level of institution. 4-year: bachelor's degree. 2-year: associate's degree.
    `control` text, -- Public,. Private not-for-profit,. Private for-profit.
    `schools_count` integer, -- No description.
    `counted_pct` text, -- counted percentage. Percentage of students in the entering class (Fall 2007 at 4-year institutions, Fall 2010 at 2-year institutions) who are first-time, full-time, degree-seeking students and generally part of the official graduation rate.
    `awards_per_state_value` text, -- awards per state value. Completions per 100 FTE (full-time equivalent) undergraduate students, state and sector average.
    `awards_per_natl_value` real, -- awards per national value. Completions per 100 FTE (full-time equivalent) undergraduate students, national sector average.
    `exp_award_state_value` text, -- expected award state value. Spending per completion, state and sector average.
    `exp_award_natl_value` integer, -- expected award national value. Spending per completion, national sector average.
    `state_appr_value` text, -- state appropriation value. State appropriations to higher education in fiscal year 2011 per resident.
    `state_appr_rank` text, -- No description.
    `grad_rate_rank` text, -- No description.
    `awards_per_rank` text, -- No description.
    primary key (stateid, level, control)
);

CREATE TABLE institution_grads (
    `unitid` integer, -- unit id. Education Unit ID number.
    `year` integer, -- year of data release.
    `gender` text, -- gender of students. 'B' = both genders; 'M' = male; 'F' = female.
    `race` text, -- race/ethnicity of students. 'X' = all students; 'Ai' = American Indian; 'A' = Asian; 'B' = Black; 'H' = Hispanic; 'W' = White.
    `cohort` text, -- degree-seeking cohort type.  '4y bach' = Bachelor's/equivalent-seeking cohort at 4-year institutions;  '4y other' = Students seeking another type of degree or certificate at a 4-year institution;  '2y all' = Degree-seeking students at 2-year institutions. 4-year degree is bachelor's degree. 2-year degree is associate's degree.
    `grad_cohort` text, -- graduation cohort. Number of first-time, full-time, degree-seeking students in the cohort being tracked, minus any exclusions.
    `grad_100` text, -- graduation 100. Number of students who graduated within 100 percent of normal/expected time.
    `grad_150` text, -- graduation 150. Number of students who graduated within 150 percent of normal/expected time.
    `grad_100_rate` text, -- Percentage of students who graduated within 100 percent of normal/expected time.
    `grad_150_rate` text, -- Percentage of students who graduated within 150 percent of normal/expected time.
    foreign key (unitid) references institution_details(unitid)
);

