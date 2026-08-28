CREATE TABLE chapters (
    `id` integer auto_increment, -- unique id number identifying the chapter.
    `Act` integer, -- An act is a major division of a theatre work, including a play, film, opera, or musical theatre. An act can consist of one or more scenes.
    `Scene` integer, -- A scene is a dramatic part of a story, at a specific time and place, between specific characters.
    `Description` text, -- textual description of the chapter.
    `work_id` integer, -- work id. id number identifying the work.
    primary key (id)
);

CREATE TABLE paragraphs (
    `id` integer , -- unique id number identifying the paragraphs.
    `ParagraphNum` integer, -- paragraph number. unique id number identifying the paragraph number.
    `PlainText` text, -- Plain Text. main content of the paragraphs.
    `character_id` integer, -- character id. unique id number identifying the mentioned character.
    `chapter_id` integer, -- chapter id. unique id number identifying the related chapter. if number of the paragraphs is > 150, then it means this is a long chapter.
    primary key (id)
);

CREATE TABLE characters (
    `id` integer, -- unique id number identifying the characters.
    `CharName` text, -- char name. character name.
    `Abbrev` text, -- abbreviation. abbreviation. An abbreviation is a shortened form of a word or phrase.
    `Description` text, -- description of the character.
    primary key (id)
);

CREATE TABLE works (
    `id` integer, -- unique id number identifying the work.
    `Title` text, -- title of the work. the short title or abbreviated title.
    `LongTitle` text, -- Long Title. full title of the work.
    `Date` integer, -- character id. date of the work.
    `GenreType` text, -- genre type. the type of the genere.
    primary key (id)
);

