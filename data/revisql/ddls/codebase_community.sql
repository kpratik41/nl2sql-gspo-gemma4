CREATE TABLE postLinks (
    `Id` integer, -- the post link id.
    `CreationDate` datetime, -- Creation Date. the creation date of the post link.
    `PostId` integer, -- Post Id. the post id.
    `RelatedPostId` integer, -- Related Post Id. the id of the related post.
    `LinkTypeId` integer, -- Link Type Id. the id of the link type.
    primary key (Id),
    foreign key (PostId) references posts (Id)
    foreign key (RelatedPostId) references posts (Id)
);

CREATE TABLE postHistory (
    `Id` integer, -- the post history id.
    `PostHistoryTypeId` integer, -- Post History Type Id. the id of the post history type.
    `PostId` integer, -- Post Id. the unique id of the post.
    `RevisionGUID` integer, -- Revision GUID. the revision globally unique id of the post.
    `CreationDate` datetime, -- Creation Date. the creation date of the post.
    `UserId` integer, -- User Id. the user who post the post.
    `Text` text, -- the detailed content of the post.
    `Comment` text, -- comments of the post.
    `UserDisplayName` text, -- User Display Name. user's display name.
    primary key (Id),
    foreign key (PostId) references posts (Id)
    foreign key (UserId) references users (Id)
);

CREATE TABLE badges (
    `Id` integer, -- the badge id.
    `UserId` integer, -- User Id. the unique id of the user.
    `Name` text, -- the badge name the user obtained.
    `Date` datetime, -- the date that the user obtained the badge.
    primary key (Id),
    foreign key (UserId) references users (Id)
);

CREATE TABLE posts (
    `Id` integer, -- the post id.
    `PostTypeId` integer, -- Post Type Id. the id of the post type.
    `AcceptedAnswerId` integer, -- Accepted Answer Id. the accepted answer id of the post.
    `CreaionDate` datetime, -- Creation Date. the creation date of the post.
    `Score` integer, -- the score of the post.
    `ViewCount` integer, -- View Count. the view count of the post. Higher view count means the post has higher popularity.
    `Body` text, -- the body of the post.
    `OwnerUserId` integer, -- Owner User Id. the id of the owner user.
    `LasActivityDate` datetime, -- Last Activity Date. the last activity date.
    `Title` text, -- the title of the post.
    `Tags` text, -- the tag of the post.
    `AnswerCount` integer, -- Answer Count. the total number of answers of the post.
    `CommentCount` integer, -- Comment Count. the total number of comments of the post.
    `FavoriteCount` integer, -- Favorite Count. the total number of favorites of the post. more favorite count refers to more valuable posts.
    `LastEditorUserId` integer, -- Last Editor User Id. the id of the last editor.
    `LastEditDate` datetime, -- Last Edit Date. the last edit date.
    `CommunityOwnedDate` datetime, -- Community Owned Date. the community owned date.
    `ParentId` integer, -- the id of the parent post. If the parent id is null, the post is the root post. Otherwise, the post is the child post of other post.
    `ClosedDate` data_format, -- Closed Date. the closed date of the post. if ClosedDate is null or empty, it means this post is not well-finished. if CloseDate is not null or empty, it means this post has well-finished.
    `OwnerDisplayName` text, -- Owner Display Name. the display name of the post owner.
    `LastEditorDisplayName` text, -- Last Editor Display Name. the display name of the last editor.
    primary key (Id),
    foreign key (LastEditorUserId) references users (Id)
    foreign key (OwnerUserId) references users (Id)
    foreign key (ParentId) references posts (Id)
);

CREATE TABLE users (
    `Id` integer, -- the user id.
    `Reputation` integer, -- the user's reputation. The user with higher reputation has more influence.
    `CreationDate` datetime, -- Creation Date. the creation date of the user account.
    `DisplayName` text, -- Display Name. the user's display name.
    `LastAccessDate` datetime, -- Last Access Date. the last access date of the user account.
    `WebsiteUrl` text, -- Website Url. the website url of the user account.
    `Location` text, -- user's location.
    `AboutMe` text, -- About Me. the self introduction of the user.
    `Views` integer, -- the number of views.
    `UpVotes` integer, -- the number of upvotes.
    `DownVotes` integer, -- the number of downvotes.
    `AccountId` integer, -- Account Id. the unique id of the account.
    `Age` integer, -- user's age.  teenager: 13-18.  adult: 19-65.  elder: > 65.
    `ProfileImageUrl` text, -- Profile Image Url. the profile image url.
    primary key (Id)
);

CREATE TABLE tags (
    `Id` integer, -- the tag id.
    `TagName` text, -- Tag Name. the name of the tag.
    `Count` integer, -- the count of posts that contain this tag. more counts --> this tag is more popular.
    `ExcerptPostId` integer, -- Excerpt Post Id. the excerpt post id of the tag.
    `WikiPostId` integer, -- Wiki Post Id. the wiki post id of the tag.
    primary key (Id),
    foreign key (ExcerptPostId) references posts (Id)
);

CREATE TABLE votes (
    `Id` integer, -- the vote id.
    `PostId` integer, -- Post Id. the id of the post that is voted.
    `VoteTypeId` integer, -- Vote Type Id. the id of the vote type.
    `CreationDate` datetime, -- Creation Date. the creation date of the vote.
    `UserId` integer, -- User Id. the id of the voter.
    `BountyAmount` integer, -- Bounty Amount. the amount of bounty.
    primary key (Id),
    foreign key (PostId) references posts (Id)
    foreign key (UserId) references users (Id)
);

CREATE TABLE comments (
    `Id` integer, -- the comment Id.
    `PostId` integer, -- Post Id. the unique id of the post.
    `Score` integer, -- rating score. The score is from 0 to 100. The score more than 60 refers that the comment is a positive comment. The score less than 60 refers that the comment is a negative comment.
    `Text` text, -- the detailed content of the comment.
    `CreationDate` datetime, -- Creation Date. the creation date of the comment.
    `UserId` integer, -- User Id. the id of the user who post the comment.
    `UserDisplayName` text, -- User Display Name. user's display name.
    primary key (Id),
    foreign key (PostId) references posts (Id)
    foreign key (UserId) references users (Id)
);

