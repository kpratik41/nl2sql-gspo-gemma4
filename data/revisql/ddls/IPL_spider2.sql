CREATE TABLE player(
        player_id INTEGER PRIMARY KEY, -- example: [1, 2, 3]
        player_name TEXT, -- example: ['SC Ganguly', 'BB McCullum', 'RT Ponting']
        dob DATE, -- example: ['1972-07-08', '1981-09-27', '1974-12-19']
        batting_hand TEXT, -- example: ['Left-hand bat', 'Right-hand bat']
        bowling_skill TEXT, -- example: ['Right-arm medium', 'Right-arm offbreak', 'Right-arm fast-medium']
        country_name TEXT -- example: ['India', 'New Zealand', 'Australia']
    )

CREATE TABLE team(
        team_id INTEGER PRIMARY KEY, -- example: [2, 3, 4]
        name TEXT -- example: ['Royal Challengers Bangalore', 'Chennai Super Kings', 'Kings XI Punjab']
    )

CREATE TABLE match(
        match_id INTEGER PRIMARY KEY, -- example: [335987, 335988, 335989]
        team_1 INTEGER, -- example: [2, 4, 6]
        team_2 INTEGER, -- example: [1, 3, 5]
        match_date DATE, -- example: ['2008-04-18', '2008-04-19', '2008-04-20']
        season_id INTEGER, -- example: [1, 2, 3]
        venue TEXT, -- example: ['M Chinnaswamy Stadium', 'Punjab Cricket Association Stadium Mohali', 'Feroz Shah Kotla']
        toss_winner INTEGER, -- example: [2, 3, 5]
        toss_decision TEXT, -- example: ['field', 'bat']
        win_type TEXT, -- example: ['runs', 'wickets']
        win_margin INTEGER, -- example: [140, 33, 9]
        outcome_type TEXT, -- example: ['Result']
        match_winner INTEGER, -- example: [1, 3, 6]
        man_of_the_match INTEGER -- example: [2, 19, 90]
    )

CREATE TABLE player_match(
        match_id INTEGER NOT NULL, -- example: [335987, 335988, 335989]
        player_id INTEGER NOT NULL, -- example: [1, 2, 3]
        role TEXT, -- example: ['Captain', 'Keeper', 'Player']
        team_id INTEGER, -- example: [7, 1, 11]
        PRIMARY KEY(match_id, player_id)
    )

CREATE TABLE ball_by_ball(
        match_id INTEGER NOT NULL, -- example: [335987, 335988, 335989]
        over_id INTEGER NOT NULL, -- example: [1, 2, 3]
        ball_id INTEGER NOT NULL, -- example: [1, 2, 3]
        innings_no INTEGER NOT NULL, -- example: [2, 1]
        team_batting INTEGER, -- example: [2, 1, 3]
        team_bowling INTEGER, -- example: [1, 2, 4]
        striker_batting_position INTEGER, -- example: [1, 2, 3]
        striker INTEGER, -- example: [6, 2, 7]
        non_striker INTEGER, -- example: [7, 1, 6]
        bowler INTEGER, -- example: [106, 14, 15]
        PRIMARY KEY(match_id, over_id, ball_id, innings_no)
    )

CREATE TABLE batsman_scored(
        match_id INTEGER NOT NULL, -- example: [335987, 335988, 335989]
        over_id INTEGER NOT NULL, -- example: [1, 2, 3]
        ball_id INTEGER NOT NULL, -- example: [1, 2, 3]
        runs_scored INTEGER, -- example: [1, 0, 4]
        innings_no INTEGER NOT NULL, -- example: [2, 1]
        PRIMARY KEY(match_id, over_id, ball_id, innings_no)
    )

CREATE TABLE wicket_taken(
        match_id INTEGER NOT NULL, -- example: [335987, 335988, 335989]
        over_id INTEGER NOT NULL, -- example: [2, 3, 5]
        ball_id INTEGER NOT NULL, -- example: [1, 2, 5]
        player_out INTEGER, -- example: [154, 46, 8]
        kind_out TEXT, -- example: ['caught', 'bowled', 'run out']
        innings_no INTEGER NOT NULL, -- example: [2, 1]
        PRIMARY KEY(match_id, over_id, ball_id, innings_no)
    )

CREATE TABLE extra_runs(
        match_id INTEGER NOT NULL, -- example: [335987, 335988, 335989]
        over_id INTEGER NOT NULL, -- example: [1, 2, 3]
        ball_id INTEGER NOT NULL, -- example: [1, 2, 3]
        extra_type TEXT, -- example: ['legbyes', 'wides', 'byes']
        extra_runs INTEGER, -- example: [1, 4, 2]
        innings_no INTEGER NOT NULL, -- example: [1, 2]
        PRIMARY KEY(match_id, over_id, ball_id, innings_no)
    )

