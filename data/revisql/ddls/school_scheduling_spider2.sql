CREATE TABLE Buildings (
    BuildingCode TEXT NOT NULL, -- example: ['AS', 'CC', 'GYM']
    BuildingName TEXT, -- example: ['Arts and Sciences', 'College Center', 'PE and Wellness']
    NumberOfFloors INTEGER, -- example: [1, 2, 3]
    ElevatorAccess BOOLEAN NOT NULL DEFAULT 0, -- example: [1, 0]
    SiteParkingAvailable BOOLEAN NOT NULL DEFAULT 0, -- example: [1, 0]
    PRIMARY KEY (BuildingCode)
)

CREATE TABLE Categories (
    CategoryID TEXT NOT NULL, -- example: ['ACC', 'ART', 'BIO']
    CategoryDescription TEXT, -- example: ['Accounting', 'Art', 'Biology']
    DepartmentID INTEGER DEFAULT 0, -- example: [1, 2, 3]
    PRIMARY KEY (CategoryID)
)

CREATE TABLE Class_Rooms (
    ClassRoomID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, -- example: [1514, 1515, 1519]
    BuildingCode TEXT, -- example: ['AS', 'CC', 'IB']
    PhoneAvailable BOOLEAN NOT NULL DEFAULT 0, -- example: [1, 0]
    FOREIGN KEY (BuildingCode) REFERENCES Buildings(BuildingCode)
)

CREATE TABLE sqlite_sequence(name,seq)

CREATE TABLE Classes (
    ClassID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, -- example: [2015, 5015, 1500]
    SubjectID INTEGER DEFAULT 0, -- example: [1, 2, 3]
    ClassRoomID INTEGER DEFAULT 0, -- example: [1131, 1142, 1231]
    Credits INTEGER DEFAULT 0, -- example: [5, 4, 3]
    StartDate DATE, -- example: ['2017-09-12', '2017-09-11', '2017-09-16']
    StartTime TIME, -- example: ['10:00:00', '15:30:00', '08:00:00']
    Duration INTEGER DEFAULT 0, -- example: [50, 110, 140]
    MondaySchedule BOOLEAN NOT NULL DEFAULT 0, -- example: [0, 1]
    TuesdaySchedule BOOLEAN NOT NULL DEFAULT 0, -- example: [1, 0]
    WednesdaySchedule BOOLEAN NOT NULL DEFAULT 0, -- example: [1, 0]
    ThursdaySchedule BOOLEAN NOT NULL DEFAULT 0, -- example: [1, 0]
    FridaySchedule BOOLEAN NOT NULL DEFAULT 0, -- example: [1, 0]
    SaturdaySchedule BOOLEAN NOT NULL DEFAULT 0, -- example: [1, 0]
    FOREIGN KEY (ClassRoomID) REFERENCES Class_Rooms(ClassRoomID),
    FOREIGN KEY (SubjectID) REFERENCES Subjects(SubjectID)
)

CREATE TABLE Departments (
    DepartmentID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, -- example: [3, 5, 2]
    DeptName TEXT, -- example: ['Business Administration', 'Sciences', 'Humanities']
    DeptChair INTEGER DEFAULT 0, -- example: [98005, 98007, 98010]
    FOREIGN KEY (DeptChair) REFERENCES Staff(StaffID)
)

CREATE TABLE Faculty (
    StaffID INTEGER NOT NULL DEFAULT 0, -- example: [98005, 98007, 98010]
    Title TEXT, -- example: ['Professor', 'Instructor', 'Associate Professor']
    Status TEXT, -- example: ['Full Time', 'On Leave', 'Part Time']
    Tenured BOOLEAN NOT NULL DEFAULT 0, -- example: [1, 0]
    PRIMARY KEY (StaffID),
    FOREIGN KEY (StaffID) REFERENCES Staff(StaffID)
)

CREATE TABLE Faculty_Categories (
    StaffID INTEGER NOT NULL, -- example: [98005, 98007, 98010]
    CategoryID TEXT NOT NULL DEFAULT 'ACC', -- example: ['ACC', 'ART', 'BIO']
    PRIMARY KEY (StaffID, CategoryID),
    FOREIGN KEY (CategoryID) REFERENCES Categories(CategoryID),
    FOREIGN KEY (StaffID) REFERENCES Faculty(StaffID)
)

CREATE TABLE Faculty_Classes (
    ClassID INTEGER NOT NULL, -- example: [1000, 1002, 1004]
    StaffID INTEGER NOT NULL, -- example: [98005, 98007, 98011]
    PRIMARY KEY (ClassID, StaffID),
    FOREIGN KEY (ClassID) REFERENCES Classes(ClassID),
    FOREIGN KEY (StaffID) REFERENCES Staff(StaffID)
)

CREATE TABLE Faculty_Subjects (
    StaffID INTEGER NOT NULL DEFAULT 0, -- example: [98005, 98007, 98010]
    SubjectID INTEGER NOT NULL DEFAULT 0, -- example: [1, 2, 3]
    ProficiencyRating REAL DEFAULT 0, -- example: [10.0, 9.0, 8.0]
    PRIMARY KEY (StaffID, SubjectID),
    FOREIGN KEY (StaffID) REFERENCES Faculty(StaffID),
    FOREIGN KEY (SubjectID) REFERENCES Subjects(SubjectID)
)

CREATE TABLE Majors (
    MajorID INTEGER NOT NULL, -- example: [1, 2, 3]
    Major TEXT, -- example: ['General Studies', 'English', 'Music']
    PRIMARY KEY (MajorID)
)

CREATE TABLE Staff (
    StaffID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, -- example: [98012, 98063, 98059]
    StfFirstName TEXT, -- example: ['Suzanne', 'Gary', 'Jeffrey']
    StfLastname TEXT, -- example: ['Viescas', 'Hallmark', 'Smith']
    StfStreetAddress TEXT, -- example: ['15127 NE 24th, #383', 'Route 2, Box 203B', '30301 - 166th Ave. N.E.']
    StfCity TEXT, -- example: ['Redmond', 'Auburn', 'Fremont']
    StfState TEXT, -- example: ['WA', 'CA', 'TX']
    StfZipCode TEXT, -- example: ['77201', '78284', '79993']
    StfAreaCode TEXT, -- example: ['425', '253', '510']
    StfPhoneNumber TEXT, -- example: ['555-2686', '555-2676', '555-2596']
    Salary REAL, -- example: [44000.0, 53000.0, 52000.0]
    DateHired DATE, -- example: ['1986-05-31', '1985-01-21', '1983-10-06']
    Position TEXT -- example: ['Faculty', 'Registrar', 'Secretary']
)

CREATE TABLE Student_Class_Status (
    ClassStatus INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, -- example: [1, 2, 3]
    ClassStatusDescription TEXT -- example: ['Enrolled', 'Completed', 'Withdrew']
)

CREATE TABLE Student_Schedules (
    StudentID INTEGER NOT NULL, -- example: [1001, 1002, 1003]
    ClassID INTEGER NOT NULL, -- example: [1000, 1168, 2907]
    ClassStatus INTEGER DEFAULT 0, -- example: [2, 1, 3]
    Grade REAL DEFAULT 0, -- example: [99.83, 70.0, 67.33]
    PRIMARY KEY (StudentID, ClassID),
    FOREIGN KEY (ClassID) REFERENCES Classes(ClassID),
    FOREIGN KEY (ClassStatus) REFERENCES Student_Class_Status(ClassStatus),
    FOREIGN KEY (StudentID) REFERENCES Students(StudentID)
)

CREATE TABLE Students (
    StudentID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, -- example: [1019, 1001, 1008]
    StudFirstName TEXT, -- example: ['Kerry', 'David', 'Betsy']
    StudLastName TEXT, -- example: ['Patterson', 'Hamilton', 'Stadick']
    StudStreetAddress TEXT, -- example: ['9877 Hacienda Drive', '908 W. Capital Way', '611 Alpine Drive']
    StudCity TEXT, -- example: ['San Antonio', 'Tacoma', 'Palm Springs']
    StudState TEXT, -- example: ['TX', 'WA', 'CA']
    StudZipCode TEXT, -- example: ['75204', '78284', '79402']
    StudAreaCode TEXT, -- example: ['206', '210', '253']
    StudPhoneNumber TEXT, -- example: ['555-2706', '555-2701', '555-2696']
    StudGPA REAL DEFAULT 0, -- example: [74.465, 78.755, 85.235]
    StudMajor INTEGER, -- example: [1, 2, 3]
    FOREIGN KEY (StudMajor) REFERENCES Majors(MajorID)
)

CREATE TABLE Subjects (
    SubjectID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, -- example: [1, 6, 7]
    CategoryID TEXT, -- example: ['ACC', 'ART', 'BIO']
    SubjectCode TEXT UNIQUE, -- example: ['ACC 210', 'ACC 220', 'ACC 230']
    SubjectName TEXT, -- example: ['Financial Accounting Fundamentals I', 'Financial Accounting Fundamentals II', 'Fundamentals of Managerial Accounting']
    SubjectPreReq TEXT DEFAULT NULL, -- example: ['ACC 210', 'ACC 220', 'BUS 170']
    SubjectDescription TEXT, -- example (truncated): ['Introduces basic accounting concepts, principles and prodcedures for recording business transactions...', 'Applications of basic accounting concepts, principles and procedures to more complex business situat...', 'Analysis of accounting data as part of the managerial process of planning, decision making and contr...']
    FOREIGN KEY (CategoryID) REFERENCES Categories(CategoryID),
    FOREIGN KEY (SubjectPreReq) REFERENCES Subjects(SubjectCode)
)

