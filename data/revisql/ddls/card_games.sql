CREATE TABLE legalities (
    `id` integer, -- unique id identifying this legality.
    `format` text, -- format of play. each value refers to different rules to play.
    `status` text, -- • legal. • banned. • restricted.
    `uuid` text, -- No description.
    primary key (id)
);

CREATE TABLE cards (
    `id` integer, -- unique id number identifying the cards.
    `artist` text, -- The name of the artist that illustrated the card art.
    `asciiName` text, -- ascii Name. The ASCII(opens new window) (Basic/128) code formatted card name with no special unicode characters.
    `availability` text, -- A list of the card's available printing types. "arena", "dreamcast", "mtgo", "paper", "shandalar".
    `borderColor` text, -- border Color. The color of the card border. "black", "borderless", "gold", "silver", "white".
    `cardKingdomFoilId` text, -- card Kingdom Foil Id. cardKingdomFoilId, when paired with cardKingdomId that is not Null, is incredibly powerful.
    `cardKingdomId` text, -- card Kingdom Id. A list of all the colors in the color indicator.
    `colorIdentity` text, -- color Identity. A list of all the colors found in manaCost, colorIndicator, and text.
    `colorIndicator` text, -- color Indicator. A list of all the colors in the color indicator (The symbol prefixed to a card's types).
    `colors` text, -- A list of all the colors in manaCost and colorIndicator. Some cards may not have values, such as cards with "Devoid" in its text.
    `convertedManaCost` real, -- converted Mana Cost. The converted mana cost of the card. Use the manaValue property. if value is higher, it means that this card cost more converted mana.
    `duelDeck` text, -- duel Deck. The indicator for which duel deck the card is in.
    `edhrecRank` integer, -- rec Rank in edh. The card rank on EDHRec.
    `faceConvertedManaCost` real, -- face Converted Mana Cost. The converted mana cost or mana value for the face for either half or part of the card. if value is higher, it means that this card cost more converted mana for the face.
    `faceName` text, -- face Name. The name on the face of the card.
    `flavorName` text, -- flavor Name. The promotional card name printed above the true card name on special cards that has no game function.
    `flavorText` text, -- flavor Text. The italicized text found below the rules text that has no game function.
    `frameEffects` text, -- frame Effects. The visual frame effects. "colorshifted", "companion", "compasslanddfc", "devoid", "draft", "etched", "extendedart", "fullart", "inverted", "legendary", "lesson", "miracle", "mooneldrazidfc", "nyxtouched", "originpwdfc", "showcase", "snow", "sunmoondfc", "textless", "tombstone", "waxingandwaningmoondfc".
    `frameVersion` text, -- frame Version. The version of the card frame style. "1993", "1997", "2003", "2015", "future".
    `hand` text, -- The starting maximum hand size total modifier. A + or - character precedes an integer. positive maximum hand size: +1, +2, .... negative maximum hand size: -1, .... neural maximum hand size: 0....
    `hasAlternativeDeckLimit` integer, -- has Alternative Deck Limit. If the card allows a value other than 4 copies in a deck. 0: disallow 1: allow.
    `hasContentWarning` integer, -- has Content Warning. If the card marked by Wizards of the Coast (opens new window) for having sensitive content. See this official article (opens new window) for more information. 0: doesn't have 1: has sensitve content or Wizards of the Coast. Cards with this property may have missing or degraded properties and values.
    `hasFoil` integer, -- has Foil. If the card can be found in foil. 0: cannot be found 1: can be found.
    `hasNonFoil` integer, -- has Non Foil. If the card can be found in non-foil. 0: cannot be found 1: can be found.
    `isAlternative` integer, -- is Alternative. If the card is an alternate variation to an original printing. 0: is not 1: is.
    `isFullArt` integer, -- is Full Art. If the card has full artwork. 0: doesn't have, 1: has full artwork.
    `isOnlineOnly` integer, -- is Online Only. If the card is only available in online game variations. 0: is not 1: is.
    `isOversized` integer, -- is Oversized. If the card is oversized. 0: is not 1: is.
    `isPromo` integer, -- is Promotion. If the card is a promotional printing. 0: is not 1: is.
    `isReprint` integer, -- is Reprint. If the card has been reprinted. 0: has not 1: has not been.
    `isReserved` integer, -- is Reserved. If the card is on the Magic: The Gathering Reserved List (opens new window). If the card is on the Magic, it will appear in The Gathering Reserved List.
    `isStarter` integer, -- is Starter. If the card is found in a starter deck such as Planeswalker/Brawl decks. 0: is not 1: is.
    `isStorySpotlight` integer, -- is Story Spotlight. If the card is a Story Spotlight card. 0: is not 1: is.
    `isTextless` integer, -- is Text less. If the card does not have a text box. 0: has a text box; 1: doesn't have a text box;
    `isTimeshifted` integer, -- is Time shifted. If the card is time shifted. If the card is "timeshifted", a feature of certain sets where a card will have a different frameVersion.
    `keywords` text, -- A list of keywords found on the card.
    `layout` text, -- The type of card layout. For a token card, this will be "token".
    `leadershipSkills` text, -- leadership Skills. A list of formats the card is legal to be a commander in.
    `life` text, -- The starting life total modifier. A plus or minus character precedes an integer.
    `loyalty` text, -- The starting loyalty value of the card. Used only on cards with "Planeswalker" in its types. empty means unkown.
    `manaCost` text, -- mana Cost. The mana cost of the card wrapped in brackets for each value. manaCost is unconverted mana cost.
    `mcmId` text, -- NOT USEFUL.
    `mcmMetaId` text, -- NOT USEFUL.
    `mtgArenaId` text, -- NOT USEFUL.
    `mtgjsonV4Id` text, -- NOT USEFUL.
    `mtgoFoilId` text, -- NOT USEFUL.
    `mtgoId` text, -- NOT USEFUL.
    `multiverseId` text, -- NOT USEFUL.
    `name` text, -- The name of the card. Cards with multiple faces, like "Split" and "Meld" cards are given a delimiter.
    `number` text, -- The number of the card.
    `originalReleaseDate` text, -- original Release Date. The original release date in ISO 8601(opens new window) format for a promotional card printed outside of a cycle window, such as Secret Lair Drop promotions.
    `originalText` text, -- original Text. The text on the card as originally printed.
    `originalType` text, -- original Type. The type of the card as originally printed. Includes any supertypes and subtypes.
    `otherFaceIds` text, -- other Face Ids. A list of card UUID's to this card's counterparts, such as transformed or melded faces.
    `power` text, -- The power of the card. ∞ means infinite power. null or * refers to unknown power.
    `printings` text, -- A list of set printing codes the card was printed in, formatted in uppercase.
    `promoTypes` text, -- promo Types. A list of promotional types for a card. "arenaleague", "boosterfun", "boxtopper", "brawldeck", "bundle", "buyabox", "convention", "datestamped", "draculaseries", "draftweekend", "duels", "event", "fnm", "gameday", "gateway", "giftbox", "gilded", "godzillaseries", "instore", "intropack", "jpwalker", "judgegift", "league", "mediainsert", "neonink", "openhouse", "planeswalkerstamped", "playerrewards", "playpromo", "premiereshop", "prerelease", "promopack", "release", "setpromo", "stamped", "textured", "themepack", "thick", "tourney", "wizardsplaynetwork".
    `purchaseUrls` text, -- purchase Urls. Links that navigate to websites where the card can be purchased.
    `rarity` text, -- The card printing rarity.
    `scryfallId` text, -- NOT USEFUL.
    `scryfallIllustrationId` text, -- NOT USEFUL.
    `scryfallOracleId` text, -- NOT USEFUL.
    `setCode` text, -- Set Code. The set printing code that the card is from.
    `side` text, -- The identifier of the card side. Used on cards with multiple faces on the same card. if this value is empty, then it means this card doesn't have multiple faces on the same card.
    `subtypes` text, -- A list of card subtypes found after em-dash.
    `supertypes` text, -- super types. A list of card supertypes found before em-dash. list of all types should be the union of subtypes and supertypes.
    `tcgplayerProductId` text, -- tcg player ProductId.
    `text` text, -- The rules text of the card.
    `toughness` text, -- The toughness of the card.
    `type` text, -- The type of the card as visible, including any supertypes and subtypes. "Artifact", "Card", "Conspiracy", "Creature", "Dragon", "Dungeon", "Eaturecray", "Elemental", "Elite", "Emblem", "Enchantment", "Ever", "Goblin", "Hero", "Instant", "Jaguar", "Knights", "Land", "Phenomenon", "Plane", "Planeswalker", "Scariest", "Scheme", "See", "Sorcery", "Sticker", "Summon", "Token", "Tribal", "Vanguard", "Wolf", "You’ll", "instant".
    `types` text, -- A list of all card types of the card, including Un‑sets and gameplay variants.
    `uuid` text, -- The universal unique identifier (v5) generated by MTGJSON. Each entry is unique. NOT USEFUL.
    `variations` text, -- No description.
    `watermark` text, -- The name of the watermark on the card.
    primary key (id)
);

CREATE TABLE rulings (
    `id` integer, -- unique id identifying this ruling.
    `date` date, -- date.
    `text` text, -- description about this ruling.
    `uuid` text, -- No description.
    primary key (id)
);

CREATE TABLE set_translations (
    `id` integer, -- unique id identifying this set.
    `language` text, -- language of this card set.
    `setCode` text, -- set code. the set code for this set.
    `translation` text, -- translation of this card set.
    primary key (id)
);

CREATE TABLE sets (
    `id` integer, -- unique id identifying this set.
    `baseSetSize` integer, -- base Set Size. The number of cards in the set.
    `block` text, -- The block name the set was in.
    `booster` text, -- A breakdown of possibilities and weights of cards in a booster pack.
    `code` text, -- The set code for the set.
    `isFoilOnly` integer, -- is Foil Only. If the set is only available in foil.
    `isForeignOnly` integer, -- is Foreign Only. If the set is available only outside the United States of America.
    `isNonFoilOnly` integer, -- is Non Foil Only. If the set is only available in non-foil.
    `isOnlineOnly` integer, -- is Online Only. If the set is only available in online game variations.
    `isPartialPreview` integer, -- is Partial Preview. If the set is still in preview (spoiled). Preview sets do not have complete data.
    `keyruneCode` text, -- keyrune Code. The matching Keyrune code for set image icons.
    `mcmId` integer, -- magic card market id. The Magic Card Marketset identifier.
    `mcmIdExtras` integer, -- magic card market ID Extras. The split Magic Card Market set identifier if a set is printed in two sets. This identifier represents the second set's identifier.
    `mcmName` text, -- magic card market name.
    `mtgoCode` text, -- magic the gathering online code. The set code for the set as it appears on Magic: The Gathering Online. if the value is null or empty, then it doesn't appear on Magic: The Gathering Online.
    `name` text, -- The name of the set.
    `parentCode` text, -- parent Code. The parent set code for set variations like promotions, guild kits, etc.
    `releaseDate` date, -- release Date. The release date in ISO 8601 format for the set.
    `tcgplayerGroupId` integer, -- tcg player Group Id. The group identifier of the set on TCGplayer.
    `totalSetSize` integer, -- total Set Size. The total number of cards in the set, including promotional and related supplemental products but excluding Alchemy modifications - however those cards are included in the set itself.
    `type` text, -- The expansion type of the set. "alchemy", "archenemy", "arsenal", "box", "commander", "core", "draft_innovation", "duel_deck", "expansion", "from_the_vault", "funny", "masterpiece", "masters", "memorabilia", "planechase", "premium_deck", "promo", "spellbook", "starter", "token", "treasure_chest", "vanguard".
    primary key (id)
);

CREATE TABLE foreign_data (
    `id` integer, -- unique id number identifying this row of data.
    `flavorText` text, -- flavor Text. The foreign flavor text of the card.
    `language` text, -- The foreign language of card.
    `multiverseid` integer, -- The foreign multiverse identifier of the card.
    `name` text, -- The foreign name of the card.
    `text` text, -- The foreign text ruling of the card.
    `type` text, -- The foreign type of the card. Includes any supertypes and subtypes.
    `uuid` text, -- No description.
    primary key (id)
);

