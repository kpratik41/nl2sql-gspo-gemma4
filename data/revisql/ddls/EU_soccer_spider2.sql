CREATE TABLE sqlite_sequence(name,seq)

CREATE TABLE "Player_Attributes" (
	`id`	INTEGER PRIMARY KEY AUTOINCREMENT, -- example: [1, 2, 3]
	`player_fifa_api_id`	INTEGER, -- example: [218353, 189615, 186170]
	`player_api_id`	INTEGER, -- example: [505942, 155782, 162549]
	`date`	TEXT, -- example: ['2016-02-18 00:00:00', '2015-11-19 00:00:00', '2015-09-21 00:00:00']
	`overall_rating`	INTEGER, -- example: [67, 62, 61]
	`potential`	INTEGER, -- example: [71, 66, 65]
	`preferred_foot`	TEXT, -- example: ['right', 'left']
	`attacking_work_rate`	TEXT, -- example: ['medium', 'high', 'low']
	`defensive_work_rate`	TEXT, -- example: ['medium', 'high', 'low']
	`crossing`	INTEGER, -- example: [49, 48, 80]
	`finishing`	INTEGER, -- example: [44, 43, 53]
	`heading_accuracy`	INTEGER, -- example: [71, 70, 58]
	`short_passing`	INTEGER, -- example: [61, 60, 71]
	`volleys`	INTEGER, -- example: [44, 43, 40]
	`dribbling`	INTEGER, -- example: [51, 50, 73]
	`curve`	INTEGER, -- example: [45, 44, 70]
	`free_kick_accuracy`	INTEGER, -- example: [39, 38, 69]
	`long_passing`	INTEGER, -- example: [64, 63, 68]
	`ball_control`	INTEGER, -- example: [49, 48, 71]
	`acceleration`	INTEGER, -- example: [60, 79, 80]
	`sprint_speed`	INTEGER, -- example: [64, 78, 82]
	`agility`	INTEGER, -- example: [59, 78, 79]
	`reactions`	INTEGER, -- example: [47, 46, 67]
	`balance`	INTEGER, -- example: [65, 90, 87]
	`shot_power`	INTEGER, -- example: [55, 54, 71]
	`jumping`	INTEGER, -- example: [58, 85, 84]
	`stamina`	INTEGER, -- example: [54, 79, 80]
	`strength`	INTEGER, -- example: [76, 56, 50]
	`long_shots`	INTEGER, -- example: [35, 34, 62]
	`aggression`	INTEGER, -- example: [71, 63, 62]
	`interceptions`	INTEGER, -- example: [70, 41, 40]
	`positioning`	INTEGER, -- example: [45, 44, 60]
	`vision`	INTEGER, -- example: [54, 53, 66]
	`penalties`	INTEGER, -- example: [48, 47, 59]
	`marking`	INTEGER, -- example: [65, 62, 76]
	`standing_tackle`	INTEGER, -- example: [69, 66, 63]
	`sliding_tackle`	INTEGER, -- example: [69, 66, 78]
	`gk_diving`	INTEGER, -- example: [6, 5, 14]
	`gk_handling`	INTEGER, -- example: [11, 10, 7]
	`gk_kicking`	INTEGER, -- example: [10, 9, 8]
	`gk_positioning`	INTEGER, -- example: [8, 7, 9]
	`gk_reflexes`	INTEGER, -- example: [8, 7, 12]
	FOREIGN KEY(`player_fifa_api_id`) REFERENCES `Player`(`player_fifa_api_id`),
	FOREIGN KEY(`player_api_id`) REFERENCES `Player`(`player_api_id`)
)

CREATE TABLE `Player` (
	`id`	INTEGER PRIMARY KEY AUTOINCREMENT, -- example: [3879, 401, 9179]
	`player_api_id`	INTEGER UNIQUE, -- example: [2625, 2752, 2768]
	`player_name`	TEXT, -- example: ['Aaron Appindangoye', 'Aaron Cresswell', 'Aaron Doran']
	`player_fifa_api_id`	INTEGER UNIQUE, -- example: [2, 6, 11]
	`birthday`	TEXT, -- example: ['1992-02-29 00:00:00', '1989-12-15 00:00:00', '1991-05-13 00:00:00']
	`height`	INTEGER, -- example: [182.88, 170.18, 172.72]
	`weight`	INTEGER -- example: [187, 146, 163]
)

CREATE TABLE `Match` (
	`id`	INTEGER PRIMARY KEY AUTOINCREMENT, -- example: [4769, 4770, 4771]
	`country_id`	INTEGER, -- example: [1, 1729, 4769]
	`league_id`	INTEGER, -- example: [1, 1729, 4769]
	`season`	TEXT, -- example: ['2008/2009', '2009/2010', '2010/2011']
	`stage`	INTEGER, -- example: [1, 10, 11]
	`date`	TEXT, -- example: ['2008-08-17 00:00:00', '2008-08-16 00:00:00', '2008-09-24 00:00:00']
	`match_api_id`	INTEGER UNIQUE, -- example: [483129, 483130, 483131]
	`home_team_api_id`	INTEGER, -- example: [9987, 10000, 9984]
	`away_team_api_id`	INTEGER, -- example: [9993, 9994, 8635]
	`home_team_goal`	INTEGER, -- example: [1, 0, 5]
	`away_team_goal`	INTEGER, -- example: [1, 0, 3]
	`home_player_X1`	INTEGER, -- example: [1, 2, 0]
	`home_player_X2`	INTEGER, -- example: [2, 4, 3]
	`home_player_X3`	INTEGER, -- example: [4, 6, 8]
	`home_player_X4`	INTEGER, -- example: [6, 8, 4]
	`home_player_X5`	INTEGER, -- example: [8, 6, 2]
	`home_player_X6`	INTEGER, -- example: [2, 6, 4]
	`home_player_X7`	INTEGER, -- example: [4, 8, 6]
	`home_player_X8`	INTEGER, -- example: [6, 2, 8]
	`home_player_X9`	INTEGER, -- example: [8, 4, 2]
	`home_player_X10`	INTEGER, -- example: [4, 6, 9]
	`home_player_X11`	INTEGER, -- example: [6, 4, 5]
	`away_player_X1`	INTEGER, -- example: [1, 2, 6]
	`away_player_X2`	INTEGER, -- example: [2, 4, 3]
	`away_player_X3`	INTEGER, -- example: [4, 6, 5]
	`away_player_X4`	INTEGER, -- example: [6, 8, 2]
	`away_player_X5`	INTEGER, -- example: [8, 6, 4]
	`away_player_X6`	INTEGER, -- example: [2, 4, 3]
	`away_player_X7`	INTEGER, -- example: [4, 6, 5]
	`away_player_X8`	INTEGER, -- example: [6, 8, 7]
	`away_player_X9`	INTEGER, -- example: [8, 2, 6]
	`away_player_X10`	INTEGER, -- example: [4, 6, 7]
	`away_player_X11`	INTEGER, -- example: [6, 4, 3]
	`home_player_Y1`	INTEGER, -- example: [1, 3, 0]
	`home_player_Y2`	INTEGER, -- example: [3, 0]
	`home_player_Y3`	INTEGER, -- example: [3, 5]
	`home_player_Y4`	INTEGER, -- example: [3, 5]
	`home_player_Y5`	INTEGER, -- example: [3, 7, 6]
	`home_player_Y6`	INTEGER, -- example: [7, 3, 6]
	`home_player_Y7`	INTEGER, -- example: [7, 6, 8]
	`home_player_Y8`	INTEGER, -- example: [7, 8, 6]
	`home_player_Y9`	INTEGER, -- example: [7, 10, 8]
	`home_player_Y10`	INTEGER, -- example: [10, 7, 8]
	`home_player_Y11`	INTEGER, -- example: [10, 11, 1]
	`away_player_Y1`	INTEGER, -- example: [1, 3]
	`away_player_Y2`	INTEGER, -- example: [3]
	`away_player_Y3`	INTEGER, -- example: [3, 7]
	`away_player_Y4`	INTEGER, -- example: [3, 5, 7]
	`away_player_Y5`	INTEGER, -- example: [3, 7, 6]
	`away_player_Y6`	INTEGER, -- example: [7, 3, 6]
	`away_player_Y7`	INTEGER, -- example: [7, 6, 8]
	`away_player_Y8`	INTEGER, -- example: [7, 8, 6]
	`away_player_Y9`	INTEGER, -- example: [7, 10, 8]
	`away_player_Y10`	INTEGER, -- example: [10, 7, 8]
	`away_player_Y11`	INTEGER, -- example: [10, 11, 8]
	`home_player_1`	INTEGER, -- example: [39890, 38327, 95597]
	`home_player_2`	INTEGER, -- example: [67950, 39580, 38292]
	`home_player_3`	INTEGER, -- example: [38788, 67958, 30692]
	`home_player_4`	INTEGER, -- example: [38312, 67959, 38435]
	`home_player_5`	INTEGER, -- example: [26235, 37112, 94462]
	`home_player_6`	INTEGER, -- example: [36393, 46004, 119117]
	`home_player_7`	INTEGER, -- example: [148286, 164732, 35412]
	`home_player_8`	INTEGER, -- example: [67898, 39631, 95609]
	`home_player_9`	INTEGER, -- example: [26916, 164352, 38246]
	`home_player_10`	INTEGER, -- example: [38801, 38423, 25957]
	`home_player_11`	INTEGER, -- example: [94289, 26502, 38419]
	`away_player_1`	INTEGER, -- example: [34480, 37937, 38252]
	`away_player_2`	INTEGER, -- example: [38388, 38293, 39156]
	`away_player_3`	INTEGER, -- example: [26458, 148313, 39151]
	`away_player_4`	INTEGER, -- example: [13423, 104411, 166554]
	`away_player_5`	INTEGER, -- example: [38389, 148314, 15652]
	`away_player_6`	INTEGER, -- example: [38798, 37202, 39145]
	`away_player_7`	INTEGER, -- example: [30949, 43158, 46890]
	`away_player_8`	INTEGER, -- example: [38253, 9307, 38947]
	`away_player_9`	INTEGER, -- example: [106013, 42153, 46881]
	`away_player_10`	INTEGER, -- example: [38383, 32690, 39158]
	`away_player_11`	INTEGER, -- example: [46552, 38782, 119118]
	`goal`	TEXT, -- example (truncated): ['<goal><value><comment>n</comment><stats><goals>1</goals><shoton>1</shoton></stats><event_incident_ty...', '<goal><value><comment>n</comment><stats><goals>1</goals><shoton>1</shoton></stats><event_incident_ty...', '<goal><value><comment>n</comment><stats><goals>1</goals><shoton>1</shoton></stats><event_incident_ty...']
	`shoton`	TEXT, -- example (truncated): ['<shoton><value><stats><blocked>1</blocked></stats><event_incident_typefk>61</event_incident_typefk><...', '<shoton><value><stats><blocked>1</blocked></stats><event_incident_typefk>61</event_incident_typefk><...', '<shoton><value><stats><blocked>1</blocked></stats><event_incident_typefk>61</event_incident_typefk><...']
	`shotoff`	TEXT, -- example (truncated): ['<shotoff><value><stats><shotoff>1</shotoff></stats><event_incident_typefk>9</event_incident_typefk><...', '<shotoff><value><stats><shotoff>1</shotoff></stats><event_incident_typefk>81</event_incident_typefk>...', '<shotoff><value><stats><shotoff>1</shotoff></stats><event_incident_typefk>9</event_incident_typefk><...']
	`foulcommit`	TEXT, -- example (truncated): ['<foulcommit><value><stats><foulscommitted>1</foulscommitted></stats><event_incident_typefk>37</event...', '<foulcommit><value><stats><foulscommitted>1</foulscommitted></stats><event_incident_typefk>37</event...', '<foulcommit><value><stats><foulscommitted>1</foulscommitted></stats><event_incident_typefk>37</event...']
	`card`	TEXT, -- example (truncated): ['<card><value><comment>y</comment><stats><ycards>1</ycards></stats><event_incident_typefk>73</event_i...', '<card />', '<card><value><comment>y</comment><stats><ycards>1</ycards></stats><event_incident_typefk>73</event_i...']
	`cross`	TEXT, -- example (truncated): ['<cross><value><stats><crosses>1</crosses></stats><event_incident_typefk>7</event_incident_typefk><el...', '<cross><value><stats><crosses>1</crosses></stats><event_incident_typefk>7</event_incident_typefk><el...', '<cross><value><stats><crosses>1</crosses></stats><event_incident_typefk>7</event_incident_typefk><el...']
	`corner`	TEXT, -- example (truncated): ['<corner><value><stats><corners>1</corners></stats><event_incident_typefk>329</event_incident_typefk>...', '<corner><value><stats><corners>1</corners></stats><event_incident_typefk>330</event_incident_typefk>...', '<corner><value><stats><corners>1</corners></stats><event_incident_typefk>329</event_incident_typefk>...']
	`possession`	TEXT, -- example (truncated): ['<possession><value><comment>56</comment><event_incident_typefk>352</event_incident_typefk><elapsed>2...', '<possession><value><comment>65</comment><event_incident_typefk>352</event_incident_typefk><elapsed>2...', '<possession><value><comment>45</comment><event_incident_typefk>352</event_incident_typefk><elapsed>2...']
	`B365H`	NUMERIC, -- example: [1.73, 1.95, 2.38]
	`B365D`	NUMERIC, -- example: [3.4, 3.2, 3.3]
	`B365A`	NUMERIC, -- example: [5, 3.6, 2.75]
	`BWH`	NUMERIC, -- example: [1.75, 1.8, 2.4]
	`BWD`	NUMERIC, -- example: [3.35, 3.3, 4]
	`BWA`	NUMERIC, -- example: [4.2, 3.95, 2.55]
	`IWH`	NUMERIC, -- example: [1.85, 1.9, 2.6]
	`IWD`	NUMERIC, -- example: [3.2, 3.1, 3.9]
	`IWA`	NUMERIC, -- example: [3.5, 2.3, 6]
	`LBH`	NUMERIC, -- example: [1.8, 1.9, 2.5]
	`LBD`	NUMERIC, -- example: [3.3, 3.2, 3.6]
	`LBA`	NUMERIC, -- example: [3.75, 3.5, 2.5]
	`PSH`	NUMERIC, -- example: [5.1, 2.48, 1.83]
	`PSD`	NUMERIC, -- example: [3.82, 3.52, 3.79]
	`PSA`	NUMERIC, -- example: [1.76, 2.96, 4.63]
	`WHH`	NUMERIC, -- example: [1.7, 1.83, 2.5]
	`WHD`	NUMERIC, -- example: [3.3, 3.25, 3.75]
	`WHA`	NUMERIC, -- example: [4.33, 3.6, 2.4]
	`SJH`	NUMERIC, -- example: [1.9, 1.95, 2.63]
	`SJD`	NUMERIC, -- example: [3.3, 4, 3.5]
	`SJA`	NUMERIC, -- example: [4, 3.8, 2.5]
	`VCH`	NUMERIC, -- example: [1.65, 2, 2.35]
	`VCD`	NUMERIC, -- example: [3.4, 3.25, 3.75]
	`VCA`	NUMERIC, -- example: [4.5, 3.25, 2.65]
	`GBH`	NUMERIC, -- example: [1.78, 1.85, 2.5]
	`GBD`	NUMERIC, -- example: [3.25, 3.2, 3.75]
	`GBA`	NUMERIC, -- example: [4, 3.75, 2.5]
	`BSH`	NUMERIC, -- example: [1.73, 1.91, 2.3]
	`BSD`	NUMERIC, -- example: [3.4, 3.25, 3.2]
	`BSA`	NUMERIC, -- example: [4.2, 3.6, 2.75]
	FOREIGN KEY(`country_id`) REFERENCES `country`(`id`),
	FOREIGN KEY(`league_id`) REFERENCES `League`(`id`),
	FOREIGN KEY(`home_team_api_id`) REFERENCES `Team`(`team_api_id`),
	FOREIGN KEY(`away_team_api_id`) REFERENCES `Team`(`team_api_id`),
	FOREIGN KEY(`home_player_1`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`home_player_2`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`home_player_3`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`home_player_4`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`home_player_5`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`home_player_6`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`home_player_7`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`home_player_8`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`home_player_9`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`home_player_10`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`home_player_11`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`away_player_1`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`away_player_2`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`away_player_3`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`away_player_4`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`away_player_5`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`away_player_6`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`away_player_7`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`away_player_8`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`away_player_9`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`away_player_10`) REFERENCES `Player`(`player_api_id`),
	FOREIGN KEY(`away_player_11`) REFERENCES `Player`(`player_api_id`)
)

CREATE TABLE `League` (
	`id`	INTEGER PRIMARY KEY AUTOINCREMENT, -- example: [1, 1729, 4769]
	`country_id`	INTEGER, -- example: [1, 1729, 4769]
	`name`	TEXT UNIQUE, -- example: ['Belgium Jupiler League', 'England Premier League', 'France Ligue 1']
	FOREIGN KEY(`country_id`) REFERENCES `country`(`id`)
)

CREATE TABLE `Country` (
	`id`	INTEGER PRIMARY KEY AUTOINCREMENT, -- example: [1, 1729, 4769]
	`name`	TEXT UNIQUE -- example: ['Belgium', 'England', 'France']
)

CREATE TABLE "Team" (
	`id`	INTEGER PRIMARY KEY AUTOINCREMENT, -- example: [31446, 1513, 31456]
	`team_api_id`	INTEGER UNIQUE, -- example: [1601, 1773, 1957]
	`team_fifa_api_id`	INTEGER, -- example: [673, 675, 15005]
	`team_long_name`	TEXT, -- example: ['KRC Genk', 'Beerschot AC', 'SV Zulte-Waregem']
	`team_short_name`	TEXT -- example: ['GEN', 'BAC', 'ZUL']
)

CREATE TABLE `Team_Attributes` (
	`id`	INTEGER PRIMARY KEY AUTOINCREMENT, -- example: [1, 2, 3]
	`team_fifa_api_id`	INTEGER, -- example: [434, 77, 614]
	`team_api_id`	INTEGER, -- example: [9930, 8485, 8576]
	`date`	TEXT, -- example: ['2010-02-22 00:00:00', '2014-09-19 00:00:00', '2015-09-10 00:00:00']
	`buildUpPlaySpeed`	INTEGER, -- example: [60, 52, 47]
	`buildUpPlaySpeedClass`	TEXT, -- example: ['Balanced', 'Fast', 'Slow']
	`buildUpPlayDribbling`	INTEGER, -- example: [48, 41, 64]
	`buildUpPlayDribblingClass`	TEXT, -- example: ['Little', 'Normal', 'Lots']
	`buildUpPlayPassing`	INTEGER, -- example: [50, 56, 54]
	`buildUpPlayPassingClass`	TEXT, -- example: ['Mixed', 'Long', 'Short']
	`buildUpPlayPositioningClass`	TEXT, -- example: ['Organised', 'Free Form']
	`chanceCreationPassing`	INTEGER, -- example: [60, 54, 70]
	`chanceCreationPassingClass`	TEXT, -- example: ['Normal', 'Risky', 'Safe']
	`chanceCreationCrossing`	INTEGER, -- example: [65, 63, 70]
	`chanceCreationCrossingClass`	TEXT, -- example: ['Normal', 'Lots', 'Little']
	`chanceCreationShooting`	INTEGER, -- example: [55, 64, 70]
	`chanceCreationShootingClass`	TEXT, -- example: ['Normal', 'Lots', 'Little']
	`chanceCreationPositioningClass`	TEXT, -- example: ['Organised', 'Free Form']
	`defencePressure`	INTEGER, -- example: [50, 47, 60]
	`defencePressureClass`	TEXT, -- example: ['Medium', 'Deep', 'High']
	`defenceAggression`	INTEGER, -- example: [55, 44, 70]
	`defenceAggressionClass`	TEXT, -- example: ['Press', 'Double', 'Contain']
	`defenceTeamWidth`	INTEGER, -- example: [45, 54, 70]
	`defenceTeamWidthClass`	TEXT, -- example: ['Normal', 'Wide', 'Narrow']
	`defenceDefenderLineClass`	TEXT, -- example: ['Cover', 'Offside Trap']
	FOREIGN KEY(`team_fifa_api_id`) REFERENCES `Team`(`team_fifa_api_id`),
	FOREIGN KEY(`team_api_id`) REFERENCES `Team`(`team_api_id`)
)

