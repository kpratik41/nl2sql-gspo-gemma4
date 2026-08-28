CREATE TABLE tags (
    `index` integer, -- index.
    `id` integer, -- release identifier which can be matched with id field in the torrents table.
    `tag` text, -- tag.
    primary key ("index")
);

CREATE TABLE torrents (
    `groupName` text, -- release title.
    `totalSnatched` integer, -- number of times the release has been downloaded.
    `artist` text, -- artist / group name.
    `groupYear` integer, -- release year.
    `releaseType` text, -- release type (e.g., album, single, mixtape).
    `groupId` integer, -- Unique release identifier from What.CD. Used to ensure no releases are duplicates.
    `id` integer, -- unique identifier (essentially an index).
    primary key (id)
);

