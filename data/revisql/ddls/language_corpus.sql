CREATE TABLE words (
    `wid` integer, -- word id. The word id of the Catalan language. The value is unique.
    `word` text, -- The word itself.
    `occurrences` integer, -- The occurrences of the specific word.
    primary key (CREATE)
);

CREATE TABLE pages (
    `pid` integer, -- page id. page id of Wikipedia about Catalan language.
    `lid` integer, -- language id. lid=1 means it's Catalan language.
    `page` integer, -- wikipedia page id.
    `revision` integer, -- wikipedia revision page id.
    `title` text, -- The title of this Catalan language Wikipedia page.
    `words` integer, -- number of different words in this page.
    primary key (CREATE)
);

CREATE TABLE biwords (
    `lid` integer, -- language id. lid=1 means it's Catalan language.
    `w1st` integer, -- word id of the first word. The word id of the first word of the biwords pair. The value is unique.
    `w2nd` integer, -- word id of the second word. The word id of the second word of the biwords pair. The value is unique.
    `occurrences` integer, -- times of this pair appears in this language/page.
    primary key (PRIMARY)
);

CREATE TABLE pages_words (
    `pid` integer, -- page id. page id of Wikipedia about Catalan language.
    `wid` integer, -- word id. The word id of the Catalan language. The value is unique.
    `occurrences` integer, -- times of this word appears into this page.
    primary key (PRIMARY)
);

CREATE TABLE langs_words (
    `lid` integer, -- language id. lid=1 means it's Catalan language.
    `wid` integer, -- word id. The word id of the Catalan language.
    `occurrences` integer, -- repetitions of this word in this language. it's INTEGER and DEFAULT is 0.
    primary key (PRIMARY)
);

CREATE TABLE langs (
    `lid` integer, -- language id. lid=1 means it's the Catalan language.
    `lang` text, -- language. language name. ca means Catalan language.
    `locale` text, -- The locale of the language.
    `pages` integer, -- total pages of Wikipedia in this language.
    `words` integer, -- total number of words in this pages.
    primary key (CREATE)
);

