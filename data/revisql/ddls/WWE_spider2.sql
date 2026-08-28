CREATE TABLE Promotions (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, -- example: [1, 230, 3211]
            name TEXT UNIQUE -- example: ['ECW', 'NXT', 'WCW']
        )

CREATE TABLE sqlite_sequence(name,seq)

CREATE TABLE Tables (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, -- example: [1, 11, 21]
            html TEXT UNIQUE, -- example (truncated): ['<table cellpadding="0" cellspacing="1">\n <tr class="head">\n  <th>\n   <a href="?order=date&amp;ppv...', '<table cellpadding="0" cellspacing="1">\n <tr class="head">\n  <th>\n   <a href="?order=date&amp;ppv...', '<table cellpadding="0" cellspacing="1">\n <tr class="head">\n  <th>\n   <a href="?order=date&amp;ppv...']
            url TEXT UNIQUE -- example: ['http://www.profightdb.com/cards/nxt-cards-pg1-no-103.html?order=&type=', 'http://www.profightdb.com/cards/nxt-cards-pg10-no-103.html?order=&type=', 'http://www.profightdb.com/cards/nxt-cards-pg100-no-103.html?order=&type=']
        )

CREATE TABLE Cards (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, -- example: [1, 2, 3]
            table_id INTEGER, -- example: [1, 11, 21]
            location_id INTEGER, -- example: [1, 24, 36]
            promotion_id INTEGER, -- example: [1, 230, 3211]
            event_date TEXT, -- example: ['1979-03-26', '1979-02-19', '1979-01-22']
            event_id INTEGER, -- example: [1, 2, 3]
            url TEXT UNIQUE, -- example: ['http://www.profightdb.com/cards/nxt/aberdeen-show-26093.html', 'http://www.profightdb.com/cards/nxt/albany-show-21800.html', 'http://www.profightdb.com/cards/nxt/albany-show-23663.html']
            info_html TEXT, -- example (truncated): ['<table border="0" width="100%">\n <tr>\n  <td align="left" height="23" width="40%">\n   <strong>\n  ...', '<table border="0" width="100%">\n <tr>\n  <td align="left" height="23" width="40%">\n   <strong>\n  ...', '<table border="0" width="100%">\n <tr>\n  <td align="left" height="23" width="40%">\n   <strong>\n  ...']
            match_html TEXT UNIQUE -- example (truncated): ['<table cellpadding="0" cellspacing="1" width="100%">\n <tr class="head">\n  <th width="3%">\n   no.\...', '<table cellpadding="0" cellspacing="1" width="100%">\n <tr class="head">\n  <th width="3%">\n   no.\...', '<table cellpadding="0" cellspacing="1" width="100%">\n <tr class="head">\n  <th width="3%">\n   no.\...']
        )

CREATE TABLE Locations (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, -- example: [1, 24, 36]
            name TEXT UNIQUE -- example: ['A Coruña, Galicia', 'Abbotsford, British Columbia', 'Aberdeen, Scotland']
        )

CREATE TABLE Events (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, -- example: [1, 2, 3]
            name TEXT UNIQUE -- example: ['1st Annual Ilio DiPaolo Memorial', '205 Live #1', '205 Live #10']
        )

CREATE TABLE Matches (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, -- example: [1, 2, 3]
            card_id INTEGER, -- example: [1, 2, 3]
            winner_id TEXT, -- example: ['1', '3', '5']
            win_type TEXT, -- example: ['def.', 'def. (CO)', 'def. (pin)']
            loser_id TEXT, -- example: ['2', '4', '6']
            match_type_id TEXT, -- example: ['1', '8', '9']
            duration TEXT, -- example: ['04:02', '04:35', '01:20']
            title_id TEXT, -- example: ['1', '9', '10']
            title_change INTEGER -- example: [0, 1]
    )

CREATE TABLE Belts (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, -- example: [1, 9, 10]
            name TEXT UNIQUE -- example: ['AAA Mega Championship', 'AWA World Heavyweight Championship', 'Cruiserweight Classic Championship WWE Cruiserweight Title']
    )

CREATE TABLE Wrestlers (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, -- example: [1, 2, 3]
            name UNIQUE -- example: ['"A Bryan Kendrick"', '"Bob Dylan"', '"Kane"']
    )

CREATE TABLE Match_Types (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, -- example: [1, 8, 9]
            name TEXT UNIQUE -- example: ['"3 stages of hell"', '"APA Invitational Bar Room Brawl"', '"Armageddon Rules"']
    )

