CREATE TABLE problem (name TEXT NOT NULL UNIQUE, path TEXT, type TEXT CHECK (type IN ("classification", "regression")), target TEXT, PRIMARY KEY (name))

CREATE TABLE eda (name TEXT, version INTEGER, feature TEXT, type TEXT, "range" BLOB, drop_user INTEGER CHECK (drop_user IN (0, 1)), drop_correlation INTEGER CHECK (drop_correlation IN (0, 1)), target INTEGER CHECK (target IN (0, 1)))

CREATE TABLE feature_importance (name TEXT, version INTEGER, step INTEGER, feature TEXT, importance NUMERIC)

CREATE TABLE solution (name TEXT, version INTEGER, correlation NUMERIC, nb_model INTEGER, nb_feature INTEGER, score NUMERIC, test_size NUMERIC, resampling INTEGER CHECK (resampling IN (0, 1)) DEFAULT (0))

CREATE TABLE model_score (name TEXT, version INTEGER, step INTEGER, model TEXT, train_score NUMERIC, test_score NUMERIC)

CREATE TABLE model_importance (name TEXT, version INTEGER, step INTEGER, model TEXT, importance NUMERIC)

CREATE TABLE model (name TEXT, version INTEGER, step INTEGER CHECK (step IN (1, 2, 3)), L1_model TEXT CHECK (L1_model IN ("regression", "tree")))

