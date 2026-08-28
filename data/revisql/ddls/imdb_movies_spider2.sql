CREATE TABLE "ERD" (
"table" TEXT, -- example: ['movie', 'genre', 'director_mapping']
  "column" TEXT, -- example: ['id', 'title', 'year']
  "Unnamed: 2" REAL, -- example: NULL
  "Unnamed: 3" REAL, -- example: NULL
  "Unnamed: 4" REAL, -- example: NULL
  "Unnamed: 5" REAL, -- example: NULL
  "Unnamed: 6" TEXT, -- example: ['role_mapping', '* movie_id', '* name_id']
  "Unnamed: 7" REAL, -- example: NULL
  "Unnamed: 8" REAL, -- example: NULL
  "Unnamed: 9" TEXT, -- example: ['genre', '* movie_id', '* genre']
  "Unnamed: 10" REAL, -- example: NULL
  "Unnamed: 11" REAL, -- example: NULL
  "Unnamed: 12" TEXT -- example: ['ratings', '* movie_id', 'avg_rating']
)

CREATE TABLE "movies" (
"id" TEXT, -- example: ['tt0012494', 'tt0038733', 'tt0361953']
  "title" TEXT, -- example: ['Der müde Tod', 'A Matter of Life and Death', 'The Nest of the Cuckoo Birds']
  "year" INTEGER, -- example: [2017, 2018, 2019]
  "date_published" TIMESTAMP, -- example: ['2017-06-09 00:00:00', '2017-12-08 00:00:00', '2017-10-16 00:00:00']
  "duration" INTEGER, -- example: [97, 104, 81]
  "country" TEXT, -- example: ['Germany', 'UK', 'USA']
  "worlwide_gross_income" TEXT, -- example: ['$ 12156', '$ 124241', '$ 8231']
  "languages" TEXT, -- example: ['German', 'English, French, Russian', 'English']
  "production_company" TEXT -- example: ['Decla-Bioscop AG', 'The Archers', 'Bert Williams Motion Pictures and Distributor']
)

CREATE TABLE "genre" (
"movie_id" TEXT, -- example: ['tt0012494', 'tt0038733', 'tt0060908']
  "genre" TEXT -- example: ['Thriller', 'Fantasy', 'Drama']
)

CREATE TABLE "director_mapping" (
"movie_id" TEXT, -- example: ['tt0038733', 'tt0060908', 'tt0069049']
  "name_id" TEXT -- example: ['nm0003836', 'nm0696247', 'nm0003606']
)

CREATE TABLE "role_mapping" (
"movie_id" TEXT, -- example: ['tt0038733', 'tt0060908', 'tt0069049']
  "name_id" TEXT, -- example: ['nm0000057', 'nm0001375', 'nm0178509']
  "category" TEXT -- example: ['actor', 'actress']
)

CREATE TABLE "names" (
"id" TEXT, -- example: ['nm0000002', 'nm0000110', 'nm0000009']
  "name" TEXT, -- example: ['Lauren Bacall', 'Kenneth Branagh', 'Richard Burton']
  "height" REAL, -- example: [174.0, 177.0, 175.0]
  "date_of_birth" TEXT, -- example: ['1924-09-16', '1960-12-10', '1925-11-10']
  "known_for_movies" TEXT -- example: ['tt3402236', 'tt4686844', 'tt1502407']
)

CREATE TABLE "ratings" (
"movie_id" TEXT, -- example: ['tt0012494', 'tt0038733', 'tt0060908']
  "avg_rating" REAL, -- example: [7.7, 8.1, 7.5]
  "total_votes" INTEGER, -- example: [4695, 17693, 3392]
  "median_rating" REAL -- example: [8.0, 7.0, 3.0]
)

