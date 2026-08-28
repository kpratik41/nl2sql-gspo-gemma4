CREATE TABLE course (
    `course_id` integer, -- course id. unique id number identifying courses.
    `courseLevel` text, -- course level.  Level_300: basic or medium undergraduate courses.  Level_400: high-level or harder undergraduate course.  Level_500: professional or master/graduate courses.
    primary key (course_id)
);

CREATE TABLE person (
    `p_id` integer, -- person ID. the unique number identifying each person.
    `professor` integer, -- whether the person is a professor. 1: professor. 0: student.
    `student` integer, -- whether the person is a student. 0: professor. 1: student.
    `hasPosition` text, -- has position. whether the person has a position in the faculty. 0: the person is not a faculty member. Common Sense evidence:. faculty_aff: affiliated faculty. faculty_eme: faculty employee.
    `inPhase` text, -- in phase. the phase of qualification the person is undergoing. 0: the person is not undergoing the phase of qualification.
    `yearsInProgram` text, -- years in program. the year of the program the person is at. 0: the person is not in any programs. Common Sense evidence:. yearX means the person is on the Xth year of the program.
    primary key (p_id)
);

CREATE TABLE taughtBy (
    `course_id` integer, -- course ID. the identification number identifying each course.
    `p_id` integer, -- person ID. the identification number identifying each person.
    primary key (course_id, p_id),
    foreign key (p_id) references person(p_id),
    foreign key (course_id) references course(course_id)
);

CREATE TABLE advisedBy (
    `p_id` integer, -- person id. id number identifying each person.
    `p_id_dummy` integer, -- person id dummy. the id number identifying the advisor.
    primary key (p_id, p_id_dummy),
    foreign key (p_id, p_id_dummy) references person (p_id, p_id)
);

