CREATE TABLE course (
    `course_id` integer, -- course id. unique id representing the course.
    `name` text, -- name of the course.
    `credit` integer, -- credit of the course. higher means more important.
    `diff` integer, -- difficulty. difficulty of the course. higher --> more difficult. smaller --> less difficult.
    primary key (course_id)
);

CREATE TABLE RA (
    `student_id` integer, -- student id. the id numbe representing each student.
    `capability` integer, -- the capability of student on research. (Evaluated by the professor). higher --> higher research ability / capability.
    `prof_id` integer, -- professor id. professor who advises this student. this value may be repetitive since one professor may advise many students in this semester. if a professor advise > 2 students in this semester, it means this professor's research work is heavy. or: this professor's popularity on research is higher.
    `salary` text, -- the salary of this student. med: average salary. high: higher salary than others. low: lower salary. free: unpaid RA.
    primary key (student_id, prof_id),
    foreign key (prof_id) references prof(prof_id),
    foreign key (student_id) references student(student_id)
);

CREATE TABLE prof (
    `prof_id` integer, -- professor id. unique id for professors.
    `gender` text, -- gender of the professor.
    `first_name` text, -- first name. the first name of the professor.
    `last_name` text, -- last name. the last name of the professor. full name: first name, last name.
    `email` text, -- email of the professor.
    `popularity` integer, -- popularity of the professor. higher --> more popular.
    `teachingability` integer, -- teaching ability. the teaching ability of the professor. higher --> more teaching ability, his / her lectures may have better quality.
    `graduate_from` text, -- graduate from. the school where the professor graduated from.
    primary key (prof_id)
);

CREATE TABLE registration (
    `course_id` integer, -- course id. the id of courses.
    `student_id` integer, -- student id. the id of students.
    `grade` text, -- the grades that the students acquire in this course.  A: excellent -- 4.  B: good -- 3.  C: fair -- 2.  D: poorly pass -- 1.  null or empty: this student fails to pass this course.  gpa of students for this semester = sum (credits x grade) / sum (credits).
    `sat` integer, -- satisfying degree. student satisfaction with the course.
    primary key (course_id, student_id),
    foreign key (course_id) references course(course_id),
    foreign key (student_id) references student(student_id)
);

CREATE TABLE student (
    `student_id` integer, -- student id. the unique id to identify students.
    `f_name` text, -- first name. the first name of the student.
    `l_name` text, -- last name. the last name of the student. full name: f_name, l_name.
    `phone_number` text, -- phone number.
    `email` text, -- email.
    `intelligence` integer, -- intelligence of the student. higher --> more intelligent.
    `gpa` real, -- graduate point average. gpa.
    `type` text, -- type of the student.  TPG: taught postgraduate student(master).  RPG: research postgraduate student (master).  UG: undergraduate student(bachelor). both TPG and RPG are students pursuing a masters degree; UG are students pursuing the bachelor degree.
    primary key (student_id)
);

