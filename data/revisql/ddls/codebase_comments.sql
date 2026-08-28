CREATE TABLE Repo (
    `Id` integer, -- unique id to identify repositories.
    `Url` text, -- github address of repository.
    `Stars` integer, -- stars this repository has received.
    `Forks` integer, -- forks that the repository has received.
    `Watchers` integer, -- how many watchers of this repository.
    `ProcessedTime` integer, -- Processed Time. how much processed time of downloading this repository.
    primary key (Id)
);

CREATE TABLE Solution (
    `Id` integer, -- unique id to identify this solution.
    `RepoId` integer, -- id of repository.
    `Path` test, -- path of this solution.
    `ProcessedTime` integer, -- processed time.
    `WasCompiled` integer, -- whether this solution needs to be compiled or not.
    primary key (Id)
);

CREATE TABLE Method (
    `Id` integer, -- unique id number. that identifies methods.
    `Name` text, -- name of methods. the second part is the task of this method. delimited by ".".
    `FullComment` text, -- Full Comment. full comment of this method.
    `Summary` text, -- summary of the method.
    `ApiCalls` text, -- Api Calls. linearized sequenced of API calls.
    `CommentIsXml` integer, -- CommentIs Xml. whether the comment is XML format or not. 0: the comment for this method is not XML format. 1: the comment for this method is XML format.
    `SampledAt` integer, -- Sampled At. the time of sampling.
    `SolutionId` integer, -- Solution Id. id number of solutions.
    `Lang` text, -- Language. language code of method.
    `NameTokenized` text, -- Name Tokenized. tokenized name.
    primary key (Id)
);

CREATE TABLE MethodParameter (
    `Id` integer, -- unique id number identifying method parameters.
    `MethodId` text, -- Method Id.
    `Type` text, -- type of the method.
    `Name` text, -- name of the method. if the name is not a word like "H", "z", it means this method is not well-discriminative.
    primary key (Id)
);

