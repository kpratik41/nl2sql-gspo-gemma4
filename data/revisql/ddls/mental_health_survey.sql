CREATE TABLE Answer (
    `AnswerText` text, -- Answer Text. The specific and detailed answer text of each question. The content is highly depend on the question.
    `SurveyID` integer, -- Survey ID. The id of each survey. The SurveyID is simply survey year i.e., 2014, 2016, 2017, 2018, 2019.
    `UserID` integer, -- User ID. The id of different user. Some questions can contain multiple answers, thus the same user can appear more than once for that QuestionID.
    `QuestionID` integer, -- Question ID. The id of different questions. Some questions can contain multiple answers, thus the same user can appear more than once for that QuestionID.
    primary key (UserID, QuestionID)
);

CREATE TABLE Question (
    `questiontext` text, -- question text. The detailed text of the question.
    `questionid` integer, -- question id. The unique id of the question. Each questiontext can only have one unique questionid.
    primary key (questionid)
);

CREATE TABLE Survey (
    `SurveyID` integer, -- Survey ID. The unique id of each survey. Each SurveyID is unique. And SurveyID is simply survey year ie 2014, 2016, 2017, 2018, 2019.
    `Description` text, -- The Description of the specific survey.
    primary key (SurveyID)
);

