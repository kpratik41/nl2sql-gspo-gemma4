CREATE TABLE current-terms (
    `address` text, -- the address of this legislator.
    `bioguide` text, -- bioguide id. The alphanumeric ID for this legislator.
    `caucus` text, -- caucus. For independents, the party that the legislator caucuses with, using the same values as the party field. Omitted if the legislator caucuses with the party indicated in the party field. When in doubt about the difference between the party and caucus fields, the party field is what displays after the legislator's name (i.e. "(D)") but the caucus field is what normally determines committee seniority. This field was added starting with terms for the 113th Congress.
    `chamber` text, -- chamber. â¢ senate. â¢ house.
    `class` real, -- class. For senators, their election class (1, 2, or 3). only senator has class, if the value is null or empty, it means this legislator is not senator.
    `contact_form` text, -- The website URL of the contact page on the legislator's official website.
    `district` real, -- district. For representatives, the district number they are serving from. if null or empty, they are not representatives.
    `end` text, -- the end of the term. end: The date the term ended (because the Congress ended or the legislator died or resigned, etc.). End dates follow the Constitutional end of a term. Since 1935, terms begin and end on January 3 at noon in odd-numbered years, and thus a term end date may also be a term start date. Prior to 1935, terms began on March 4 and ended either on March 3 or March 4. The end date is the last date on which the legislator served this term. Unlike the start date, whether Congress was in session or not does not affect the value of this field.
    `fax` text, -- The fax number of the legislator's Washington, D.C. office. only valid if the term is current.
    `last` text, -- the last known number.
    `name` text, -- not useful.
    `office` text, -- office. only valid if the term is current, otherwise the last known office.
    `party` text, -- The political party of the legislator. If the legislator changed parties, this is the most recent party held during the term and party_affiliations will be set. Values are typically "Democrat", "Independent", or "Republican". The value typically matches the political party of the legislator on the ballot in his or her last election, although for state affiliate parties such as "Democratic Farmer Labor" we will use the national party name ("Democrat") instead to keep the values of this field normalized.
    `party_affiliations` text, -- party affiliations. This field is present if the legislator changed party or caucus affiliation during the term. The value is a list of time periods, with start and end dates, each of which has a party field and a caucus field if applicable, with the same meanings as the main party and caucus fields. The time periods cover the entire term, so the first start will match the term start, the last end will match the term end, and the last party (and caucus if present) will match the term party (and caucus).
    `phone` text, -- The phone number of the legislator's Washington, D.C. office. only valid if the term is current, otherwise the last known number.
    `relation` text, -- not useful.
    `rss_url` text, -- Really Simple Syndication URL. The URL to the official website's RSS feed.
    `start` text, -- The date legislative service began: the date the legislator was sworn in, if known, or else the beginning of the legislator's term. Since 1935 regularly elected terms begin on January 3 at noon on odd-numbered years, but when Congress does not first meet on January 3, term start dates might reflect that swearing-in occurred on a later date. (Prior to 1935, terms began on March 4 of odd-numbered years, see here.).
    `state` text, -- state code. AK: Alaska. AL: Alabama. AR: Arkansas. AZ: Arizona. CA: California. CO: Colorado. CT: Connecticut. DE: Delaware. FL: Florida. GA: Georgia. HI: Hawaii. IA: Iowa. ID: Idaho. IL: Illinois. IN: Indiana. KS: Kansas. KY: Kentucky. LA: Louisiana. MA: Massachusetts. MD: Maryland. ME: Maine. MI: Michigan. MN: Minnesota. MO: Missouri. MS: Mississippi. MT: Montana. NC: North Carolina. ND: North Dakota. NE: Nebraska. NH: New Hampshire. NJ: New Jersey. 9 divisions of states in us: (please mention). https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf.
    `state_rank` text, -- whether they are the "junior" or "senior" senator. only valid if the term is current, otherwise the senator's rank at the time the term ended. only senator has this value.
    `title` text, -- title of the legislator.
    `type` text, -- The type of the term. Either "sen" for senators or "rep" for representatives and delegates to the House.
    `url` text, -- The official website URL of the legislator. only valid if the term is current.
    primary key (bioguide, end),
    foreign key (bioguide) references current(bioguide_id)
);

CREATE TABLE social-media (
    `bioguide` text, -- The unique alphanumeric ID for this legislator.
    `facebook` text, -- The username of the current official Facebook presence of the legislator.
    `facebook_id` real, -- The numeric ID of the current official Facebook presence of the legislator.
    `govtrack` real, -- The numeric ID for this legislator on GovTrack.us.
    `instagram` text, -- The current official Instagram handle of the legislator.
    `instagram_id` real, -- The numeric ID of the current official Instagram handle of the legislator.
    `thomas` integer, -- The numeric ID for this legislator on http://thomas.gov and http://beta.congress.gov.
    `twitter` text, -- The current official Twitter handle of the legislator.
    `twitter_id` real, -- The numeric ID of the current official twitter handle of the legislator.
    `youtube` text, -- The current official YouTube username of the legislator.
    `youtube_id` text, -- The current official YouTube channel ID of the legislator.
    primary key (bioguide),
    foreign key (bioguide) references current(bioguide_id)
);

CREATE TABLE historical (
    `ballotpedia_id` text, -- ballotpedia id. The ballotpedia.org page name for the person (spaces are given as spaces, not underscores). if this value is null or empty, it means this legislator doesn't have account on ballotpedia.org.
    `bioguide_previous_id` text, -- bioguide previous id. The previous alphanumeric ID for this legislator.
    `bioguide_id` text, -- bioguide id. The alphanumeric ID for this legislator.
    `birthday_bio` text, -- birthday bio. The legislator's birthday,. in YYYY-MM-DD format.
    `cspan_id` text, -- cspan id. The numeric ID for this legislator on C-SPAN's video website,. if this value is null or empty, it means this legislator doesn't have account on C-SPAN's video website.
    `fec_id` text, -- fec id. A list of IDs for this legislator in Federal Election Commission data. if this value is null or empty, it means this legislator hasn't not been registered in Federal Election Commission data.
    `first_name` text, -- first name. first name of the legislator.
    `gender_bio` text, -- gender bio. gender of the legislator.
    `google_entity_id_id` text, -- google entity id. if this value is null or empty, it means this legislator are not google entities.
    `govtrack_id` integer, -- govtrack id. The numeric ID for this legislator on GovTrack.us. if this value is null or empty, it means this legislator doesn't have account on GovTrack.us.
    `house_history_alternate_id` text, -- house history alternate id. The alternative numeric ID for this legislator.
    `house_history_id` real, -- house history id. The numeric ID for this legislator on http://history.house.gov/People/Search/. The ID is present only for members who have served in the U.S. House.
    `icpsr_id` real, -- interuniversity consortium for political and social research id. The numeric ID for this legislator in Keith Poole's VoteView.com website, originally based on an ID system by the Interuniversity Consortium for Political and Social Research (stored as an integer). if this value is null or empty, it means this legislator doesn't have account on VoteView.com.
    `last_name` text, -- last name. last name of the legislator.
    `lis_id` text, -- legislator id. The alphanumeric ID for this legislator found in Senate roll call votes. The ID is present only for members who attended in Senate roll call votes.
    `maplight_id` text, -- maplight id. The numeric ID for this legislator on maplight.org. if this value is null or empty, it means this legislator doesn't have account on maplight.org.
    `middle_name` text, -- middle name. the middle name of the legislator.
    `nickname_name` text, -- nickname. nickname of the legislator.
    `official_full_name` text, -- official full name.
    `opensecrets_id` text, -- opensecrets id. The alphanumeric ID for this legislator on OpenSecrets.org. if this value is null or empty, it means this legislator doesn't have account on OpenSecrets.org.
    `religion_bio` text, -- religion bio. The legislator's religion.
    `suffix_name` text, -- suffix name.
    `thomas_id` text, -- thomas id. The numeric ID for this legislator on http://thomas.gov and http://beta.congress.gov. if this value is null or empty, it means this legislator doesn't have account on both http://thomas.gov and http://beta.congress.gov.
    `votesmart_id` text, -- votesmart id. The numeric ID for this legislator on VoteSmart.org. if this value is null or empty, it means this legislator doesn't have account on VoteSmart.org.
    `wikidata_id` text, -- wikidata id. the id for wikidata.
    `wikipedia_id` text, -- wikipedia id. The Wikipedia page name for the person. if a legislator has wikipedia id, it means he or she is famous or impact.
    primary key (bioguide_id)
);

CREATE TABLE historical-terms (
    `bioguide` text, -- bioguide id. The alphanumeric ID for this legislator.
    `address` text, -- the address of this legislator.
    `chamber` text, -- chamber. â¢ senate. â¢ house.
    `class` real, -- class. For senators, their election class (1, 2, or 3). only senator has class, if the value is null or empty, it means this legislator is not senator.
    `contact_form` text, -- The website URL of the contact page on the legislator's official website.
    `district` real, -- district. For representatives, the district number they are serving from. if null or empty, they are not representatives.
    `end` text, -- the end of the term. end: The date the term ended (because the Congress ended or the legislator died or resigned, etc.). End dates follow the Constitutional end of a term. Since 1935, terms begin and end on January 3 at noon in odd-numbered years, and thus a term end date may also be a term start date. Prior to 1935, terms began on March 4 and ended either on March 3 or March 4. The end date is the last date on which the legislator served this term. Unlike the start date, whether Congress was in session or not does not affect the value of this field.
    `fax` text, -- The fax number of the legislator's Washington, D.C. office. only valid if the term is current.
    `last` text, -- the last known number.
    `middle` text, -- No description.
    `name` text, -- not useful.
    `office` text, -- office. only valid if the term is current, otherwise the last known office.
    `party` text, -- The political party of the legislator. If the legislator changed parties, this is the most recent party held during the term and party_affiliations will be set. Values are typically "Democrat", "Independent", or "Republican". The value typically matches the political party of the legislator on the ballot in his or her last election, although for state affiliate parties such as "Democratic Farmer Labor" we will use the national party name ("Democrat") instead to keep the values of this field normalized.
    `party_affiliations` text, -- party affiliations. This field is present if the legislator changed party or caucus affiliation during the term. The value is a list of time periods, with start and end dates, each of which has a party field and a caucus field if applicable, with the same meanings as the main party and caucus fields. The time periods cover the entire term, so the first start will match the term start, the last end will match the term end, and the last party (and caucus if present) will match the term party (and caucus).
    `phone` text, -- The phone number of the legislator's Washington, D.C. office. only valid if the term is current, otherwise the last known number.
    `relation` text, -- not useful.
    `rss_url` text, -- Really Simple Syndication URL. The URL to the official website's RSS feed.
    `start` text, -- The date legislative service began: the date the legislator was sworn in, if known, or else the beginning of the legislator's term. Since 1935 regularly elected terms begin on January 3 at noon on odd-numbered years, but when Congress does not first meet on January 3, term start dates might reflect that swearing-in occurred on a later date. (Prior to 1935, terms began on March 4 of odd-numbered years, see here.).
    `state` text, -- state code. AK: Alaska. AL: Alabama. AR: Arkansas. AZ: Arizona. CA: California. CO: Colorado. CT: Connecticut. DE: Delaware. FL: Florida. GA: Georgia. HI: Hawaii. IA: Iowa. ID: Idaho. IL: Illinois. IN: Indiana. KS: Kansas. KY: Kentucky. LA: Louisiana. MA: Massachusetts. MD: Maryland. ME: Maine. MI: Michigan. MN: Minnesota. MO: Missouri. MS: Mississippi. MT: Montana. NC: North Carolina. ND: North Dakota. NE: Nebraska. NH: New Hampshire. NJ: New Jersey. 9 divisions of states in us: (please mention). https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf.
    `state_rank` text, -- whether they are the "junior" or "senior" senator. only valid if the term is current, otherwise the senator's rank at the time the term ended. only senator has this value.
    `title` text, -- title of the legislator.
    `type` text, -- The type of the term. Either "sen" for senators or "rep" for representatives and delegates to the House.
    `url` text, -- The official website URL of the legislator. only valid if the term is current.
    primary key (bioguide),
    foreign key (bioguide) references historical(bioguide_id)
);

CREATE TABLE current (
    `ballotpedia_id` text, -- ballotpedia id. The ballotpedia.org page name for the person (spaces are given as spaces, not underscores). if this value is null or empty, it means this legislator doesn't have account on ballotpedia.org.
    `bioguide_id` text, -- bioguide id. The alphanumeric ID for this legislator.
    `birthday_bio` date, -- birthday bio. The legislator's birthday,. in YYYY-MM-DD format.
    `cspan_id` real, -- cspan id. The numeric ID for this legislator on C-SPAN's video website,. if this value is null or empty, it means this legislator doesn't have account on C-SPAN's video website.
    `fec_id` text, -- fec id. A list of IDs for this legislator in Federal Election Commission data. if this value is null or empty, it means this legislator hasn't not been registered in Federal Election Commission data.
    `first_name` text, -- first name. first name of the legislator.
    `gender_bio` text, -- gender bio. gender of the legislator.
    `google_entity_id_id` text, -- google entity id. if this value is null or empty, it means this legislator are not google entities.
    `govtrack_id` integer, -- govtrack id. The numeric ID for this legislator on GovTrack.us. if this value is null or empty, it means this legislator doesn't have account on GovTrack.us.
    `house_history_id` real, -- house history id. The numeric ID for this legislator on http://history.house.gov/People/Search/. The ID is present only for members who have served in the U.S. House.
    `icpsr_id` real, -- interuniversity consortium for political and social research id. The numeric ID for this legislator in Keith Poole's VoteView.com website, originally based on an ID system by the Interuniversity Consortium for Political and Social Research (stored as an integer). if this value is null or empty, it means this legislator doesn't have account on VoteView.com.
    `last_name` text, -- last name. last name of the legislator.
    `lis_id` text, -- legislator id. The alphanumeric ID for this legislator found in Senate roll call votes. The ID is present only for members who attended in Senate roll call votes.
    `maplight_id` real, -- maplight id. The numeric ID for this legislator on maplight.org. if this value is null or empty, it means this legislator doesn't have account on maplight.org.
    `middle_name` text, -- middle name. the middle name of the legislator.
    `nickname_name` text, -- nickname. nickname of the legislator.
    `official_full_name` text, -- official full name.
    `opensecrets_id` text, -- opensecrets id. The alphanumeric ID for this legislator on OpenSecrets.org. if this value is null or empty, it means this legislator doesn't have account on OpenSecrets.org.
    `religion_bio` text, -- religion bio. The legislator's religion.
    `suffix_name` text, -- suffix name.
    `thomas_id` integer, -- thomas id. The numeric ID for this legislator on http://thomas.gov and http://beta.congress.gov. if this value is null or empty, it means this legislator doesn't have account on both http://thomas.gov and http://beta.congress.gov.
    `votesmart_id` real, -- votesmart id. The numeric ID for this legislator on VoteSmart.org. if this value is null or empty, it means this legislator doesn't have account on VoteSmart.org.
    `wikidata_id` text, -- wikidata id. the id for wikidata.
    `wikipedia_id` text, -- wikipedia id. The Wikipedia page name for the person. if a legislator has wikipedia id, it means he or she is famous or impact.
    primary key (bioguide_id, cspan_id)
);

