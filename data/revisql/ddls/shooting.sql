CREATE TABLE subjects (
    `case_number` text, -- case number.
    `race` text, -- No description.
    `gender` text, -- gender. M: male; F: female.
    `last_name` text, -- last name.
    `first_name` text, -- first name.
    `full_name` text, -- full name.
    foreign key (case_number) references incidents (case_number)
);

CREATE TABLE incidents (
    `case_number` text, -- case number.
    `date` date, -- date.
    `location` text, -- location.
    `subject_statuses` text, -- subject statuses. statuses of the victims.
    `subject_weapon` text, -- subject weapon.
    `subjects` text, -- subjects.
    `subject_count` integer, -- subject_count.
    `officers` text, -- officers.
    primary key (case_number)
);

CREATE TABLE officers (
    `case_number` text, -- case number.
    `race` text, -- No description.
    `gender` text, -- gender. M: male; F: female.
    `last_name` text, -- last name.
    `first_name` text, -- first name.
    `full_name` text, -- full name.
    foreign key (case_number) references incidents (case_number)
);

