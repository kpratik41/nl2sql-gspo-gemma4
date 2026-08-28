CREATE TABLE enlist (
    `name` text, -- the name of the enlisted students.
    `organ` text, -- organization. the organization that the student enlisted in.
    foreign key ("name") references person ("name")
);

CREATE TABLE enrolled (
    `name` text, -- No description.
    `school` text, -- No description.
    `month` integer, -- No description.
    PRIMARY KEY (name,school),
    FOREIGN KEY (name) REFERENCES person (name) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE unemployed (
    `name` text, -- student who is unemployed.
    primary key ("name"),
    foreign key ("name") references person ("name")
);

CREATE TABLE disabled (
    `name` text, -- the name of the disabled students.
    primary key ("name"),
    foreign key ("name") references person ("name")
);

CREATE TABLE person (
    `name` text, -- student's name.
    primary key ("name")
);

CREATE TABLE longest_absense_from_school (
    `name` text, -- student's name.
    `month` integer, -- the duration of absence. 0 means that the student has never been absent.
    primary key ("name"),
    foreign key ("name") references person ("name")
);

CREATE TABLE male (
    `name` text, -- student's name who are male. the students who are not in this list are female.
    primary key ("name"),
    foreign key ("name") references person ("name")
);

CREATE TABLE bool (
    `name` text, -- No description.
    primary key ("name")
);

CREATE TABLE filed_for_bankrupcy (
    `name` text, -- student name who filed for bankruptcy.
    primary key ("name"),
    foreign key ("name") references person ("name")
);

CREATE TABLE no_payment_due (
    `name` text, -- student's name.
    `bool` text, -- whether the student has payment dues.  neg: the student doesn't have payment due.  pos: the student has payment due.
    primary key ("name"),
    foreign key ("name") references person ("name")
    foreign key (bool) references bool ("name")
);

