CREATE TABLE "Movie" (
"index" INTEGER, -- example: [0, 1, 2]
  "MID" TEXT, -- example: ['tt2388771', 'tt5164214', 'tt1365519']
  "title" TEXT, -- example: ['Mowgli', 'Ocean''s Eight', 'Tomb Raider']
  "year" TEXT, -- example: ['2018', '2012', '2016']
  "rating" REAL, -- example: [6.6, 6.2, 6.4]
  "num_votes" INTEGER -- example: [21967, 110861, 142585]
)

CREATE TABLE "Genre" (
"index" INTEGER, -- example: [0, 1, 2]
  "Name" TEXT, -- example: ['Adventure, Drama, Fantasy            ', 'Action, Comedy, Crime            ', 'Action, Adventure, Fantasy            ']
  "GID" INTEGER -- example: [0, 1, 2]
)

CREATE TABLE "Language" (
"index" INTEGER, -- example: [0, 1, 2]
  "Name" TEXT, -- example: ['English', 'Marathi', 'Hindi']
  "LAID" INTEGER -- example: [0, 1, 2]
)

CREATE TABLE "Country" (
"index" INTEGER, -- example: [0, 1, 2]
  "Name" TEXT, -- example: ['UK', 'USA', 'India']
  "CID" INTEGER -- example: [0, 1, 2]
)

CREATE TABLE "Location" (
"index" INTEGER, -- example: [0, 1, 2]
  "Name" TEXT, -- example: ['Durban, South Africa', 'New York City, New York, USA', 'Cape Town Film Studios, Cape Town, Western Cape, South Africa']
  "LID" INTEGER -- example: [0, 1, 2]
)

CREATE TABLE "M_Location" (
"index" INTEGER, -- example: [0, 1, 2]
  "MID" TEXT, -- example: ['tt2388771', 'tt5164214', 'tt1365519']
  "LID" REAL, -- example: [0.0, 1.0, 2.0]
  "ID" INTEGER -- example: [0, 1, 2]
)

CREATE TABLE "M_Country" (
"index" INTEGER, -- example: [0, 1, 2]
  "MID" TEXT, -- example: ['tt2388771', 'tt5164214', 'tt1365519']
  "CID" REAL, -- example: [0.0, 1.0, 2.0]
  "ID" INTEGER -- example: [0, 1, 2]
)

CREATE TABLE "M_Language" (
"index" INTEGER, -- example: [0, 1, 2]
  "MID" TEXT, -- example: ['tt2388771', 'tt5164214', 'tt1365519']
  "LAID" INTEGER, -- example: [0, 1, 2]
  "ID" INTEGER -- example: [0, 1, 2]
)

CREATE TABLE "M_Genre" (
"index" INTEGER, -- example: [0, 1, 2]
  "MID" TEXT, -- example: ['tt2388771', 'tt5164214', 'tt1365519']
  "GID" INTEGER, -- example: [0, 1, 2]
  "ID" INTEGER -- example: [0, 1, 2]
)

CREATE TABLE "Person" (
"index" INTEGER, -- example: [0, 1, 2]
  "PID" TEXT, -- example: ['nm0000288', 'nm0000949', 'nm1212722']
  "Name" TEXT, -- example: [' Christian Bale', ' Cate Blanchett', ' Benedict Cumberbatch']
  "Gender" TEXT -- example: ['Male', 'Female']
)

CREATE TABLE "M_Producer" (
"index" INTEGER, -- example: [0, 1, 2]
  "MID" TEXT, -- example: ['tt2388771', 'tt5164214', 'tt1365519']
  "PID" TEXT, -- example: [' nm0057655', ' nm0147080', ' nm0389414']
  "ID" INTEGER -- example: [0, 1, 2]
)

CREATE TABLE "M_Director" (
"index" INTEGER, -- example: [0, 1, 2]
  "MID" TEXT, -- example: ['tt2388771', 'tt5164214', 'tt1365519']
  "PID" TEXT, -- example: ['nm0785227', 'nm0002657', 'nm1012385']
  "ID" INTEGER -- example: [0, 1, 2]
)

CREATE TABLE "M_Cast" (
"index" INTEGER, -- example: [0, 1, 2]
  "MID" TEXT, -- example: ['tt2388771', 'tt5164214', 'tt1365519']
  "PID" TEXT, -- example: [' nm0000288', ' nm0000949', ' nm1212722']
  "ID" INTEGER -- example: [0, 1, 2]
)

