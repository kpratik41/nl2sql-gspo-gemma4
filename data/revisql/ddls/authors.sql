CREATE TABLE Author (
    `Id` integer, -- Id of the author.
    `Name` text, -- Author Name.
    `Affiliation` text, -- Organization name with which the author is affiliated. the name of an organization with which an author can be affiliated.
    primary key (Id)
);

CREATE TABLE Journal (
    `Id` integer, -- Journal Id.
    `ShortName` text, -- Short name.
    `FullName` text, -- Full name.
    `HomePage` text, -- Homepage URL of journal.
    primary key (Id)
);

CREATE TABLE Paper (
    `Id` integer, -- Id of the paper.
    `Title` text, -- Title of the paper.
    `Year` integer, -- Year of the paper. commonsense reasoning: if the year is "0", it means this paper is preprint, or not published.
    `ConferenceId` integer, -- Conference Id in which paper was published.
    `JournalId` integer, -- Journal Id in which paper was published. commonsense reasoning: If a paper contain "0" in both ConferenceID and JournalId, it means this paper is preprint.
    `Keyword` text, -- Keywords of the paper. commonsense reasoning: Keywords should contain words and phrases that suggest what the topic is about. Similar keywords represent similar fields or sub-field.
    primary key (Id),
    foreign key (ConferenceId) references Conference(Id),
    foreign key (JournalId) references Journal(Id)
);

CREATE TABLE Conference (
    `Id` integer, -- Conference Id.
    `ShortName` text, -- Short name.
    `FullName` text, -- Full name.
    `HomePage` text, -- Homepage URL of conference.
    primary key (Id)
);

CREATE TABLE PaperAuthor (
    `PaperId` integer, -- Paper Id.
    `AuthorId` integer, -- Author Id. commonsense reasoning: A paper can have more than one author. Co-authorship can be derived from (paper ID, author ID) pair.
    `Name` text, -- Author Name (as written on paper).
    `Affiliation` text, -- Author Affiliation (as written on paper). the name of an organization with which an author can be affiliated.
    foreign key (PaperId) references Paper(Id),
    foreign key (AuthorId) references Author(Id)
);

