CREATE TABLE university_ranking_year (
    `university_id` integer, -- university id. the id of the university.
    `ranking_criteria_id` integer, -- ranking criteria id. the id of the ranking criteria.
    `year` integer, -- ranking year.
    `score` integer, -- ranking score.
    foreign key (ranking_criteria_id) references ranking_criteria(id),
    foreign key (university_id) references university(id)
);

CREATE TABLE university_year (
    `university_id` integer, -- university id. id of the university.
    `year` integer, -- No description.
    `num_students` integer, -- number of students. the total number of students for the year.
    `student_staff_ratio` real, -- student staff ratio. student_staff_ratio = number of students / number of staff.
    `pct_international_students` real, -- pct internation student. the percentage of international students among all students. pct_international_student = number of interbational students / number of students.
    `pct_female_students` real, -- pct female students. the percentage of female students. pct_female_students = number of female students / number of students.
    foreign key (university_id) references university(id)
);

CREATE TABLE ranking_system (
    `id` integer, -- unique id number identifying ranking system.
    `system_name` text, -- system name. id number identifying ranking system.
    primary key (id)
);

CREATE TABLE ranking_criteria (
    `id` integer, -- unique id number identifying ranking criteria.
    `ranking_system_id` integer, -- ranking system id. id number identifying ranking system.
    `criteria_name` text, -- criteria name. name of the criteria.
    primary key (id),
    foreign key (ranking_system_id) references ranking_system(id)
);

CREATE TABLE university (
    `id` integer, -- unique id number identifying university.
    `country_id` text, -- country id. the country where the university locates.
    `university_name` text, -- university name. name of the university.
    primary key (id),
    foreign key (country_id) references country(id)
);

CREATE TABLE country (
    `id` integer, -- unique id number identifying country.
    `country_name` text, -- country name. the name of the country.
    primary key (id)
);

