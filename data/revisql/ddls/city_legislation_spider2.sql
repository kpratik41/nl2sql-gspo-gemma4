CREATE TABLE "aliens_details" (
"detail_id" INTEGER, -- example: [1, 2, 3]
  "favorite_food" TEXT, -- example: ['White-faced tree rat', 'Lizard, goanna', 'Indian red admiral']
  "feeding_frequency" TEXT, -- example: ['Weekly', 'Seldom', 'Often']
  "aggressive" INTEGER -- example: [1, 0]
)

CREATE TABLE "skills_dim" (
"skill_id" INTEGER, -- example: [195, 201, 164]
  "skills" TEXT, -- example: ['sharepoint', 'alteryx', 'asp.netcore']
  "type" TEXT -- example: ['analyst_tools', 'webframeworks', 'programming']
)

CREATE TABLE "legislators_terms" (
"id_bioguide" TEXT, -- example: ['B000944', 'C000127', 'C000141']
  "term_number" INTEGER, -- example: [0, 1, 2]
  "term_id" TEXT, -- example: ['B000944-0', 'C000127-0', 'C000141-0']
  "term_type" TEXT, -- example: ['rep', 'sen']
  "term_start" TEXT, -- example: ['1993-01-05', '1987-01-06', '1983-01-03']
  "term_end" TEXT, -- example: ['1995-01-03', '1989-01-03', '1985-01-03']
  "state" TEXT, -- example: ['OH', 'WA', 'MD']
  "district" REAL, -- example: [13.0, 1.0, 3.0]
  "class" REAL, -- example: [1.0, 2.0, 3.0]
  "party" TEXT, -- example: ['Democrat', 'Republican', 'Independent']
  "how" TEXT, -- example: ['appointment']
  "url" TEXT, -- example: ['http://casey.senate.gov/', 'https://fulcher.house.gov', 'http://klobuchar.senate.gov/']
  "address" TEXT, -- example: ['393 RUSSELL SENATE OFFICE BUILDING WASHINGTON DC 20510', '1520 Longworth House Office Building; Washington DC 20515-1201', '302 HART SENATE OFFICE BUILDING WASHINGTON DC 20510']
  "phone" TEXT, -- example: ['202-224-6324', '202-225-6611', '202-224-3244']
  "fax" TEXT, -- example: ['202-228-0604', '202-228-2186', '202-224-8594']
  "contact_form" TEXT, -- example: ['http://www.casey.senate.gov/contact/', 'http://www.klobuchar.senate.gov/emailamy.cfm', 'http://www.tester.senate.gov/Contact/index.cfm']
  "office" TEXT, -- example: ['393 Russell Senate Office Building', '1520 Longworth House Office Building', '302 Hart Senate Office Building']
  "state_rank" TEXT, -- example: ['senior', 'junior']
  "rss_url" TEXT, -- example: ['http://www.merkley.senate.gov/rss/', 'http://www.shaheen.senate.gov/rss/', 'http://www.warner.senate.gov/public/?a=rss.feed']
  "caucus" TEXT -- example: ['Democrat']
)

CREATE TABLE "cities_currencies" (
"currency_id" INTEGER, -- example: [1, 2, 3]
  "country_code_2" TEXT, -- example: ['af', 'al', 'dz']
  "currency_name" TEXT, -- example: ['afghani', 'lek', 'algerian dinar']
  "currency_code" TEXT -- example: ['afn', 'all', 'dzd']
)

CREATE TABLE "legislators" (
"full_name" TEXT, -- example: ['Sherrod Brown', 'Maria Cantwell', 'Benjamin L. Cardin']
  "first_name" TEXT, -- example: ['Sherrod', 'Maria', 'Benjamin']
  "last_name" TEXT, -- example: ['Brown', 'Cantwell', 'Cardin']
  "middle_name" TEXT, -- example: ['L.', 'Richard', 'P.']
  "nickname" TEXT, -- example: ['Bob', 'Bernie', 'Jim']
  "suffix" TEXT, -- example: ['Jr.', 'III', 'II']
  "other_names_end" TEXT, -- example: ['1846-01-12', '1995-09-03', '1995-01-03']
  "other_names_middle" REAL, -- example: NULL
  "other_names_last" TEXT, -- example: ['Menendez', 'Levy', 'Long']
  "birthday" TEXT, -- example: ['1952-11-09', '1958-10-13', '1943-10-05']
  "gender" TEXT, -- example: ['M', 'F']
  "id_bioguide" TEXT, -- example: ['B000944', 'C000127', 'C000141']
  "id_bioguide_previous_0" TEXT, -- example: ['F000246', 'L000266', 'W000790']
  "id_govtrack" INTEGER, -- example: [400050, 300018, 400064]
  "id_icpsr" REAL, -- example: [29389.0, 39310.0, 15408.0]
  "id_wikipedia" TEXT, -- example: ['Sherrod Brown', 'Maria Cantwell', 'Ben Cardin']
  "id_wikidata" TEXT, -- example: ['Q381880', 'Q22250', 'Q723295']
  "id_google_entity_id" TEXT, -- example: ['kg:/m/034s80', 'kg:/m/01x68t', 'kg:/m/025k3k']
  "id_house_history" REAL, -- example: [9996.0, 10608.0, 10629.0]
  "id_house_history_alternate" REAL, -- example: [13283.0]
  "id_thomas" REAL, -- example: [136.0, 172.0, 174.0]
  "id_cspan" REAL, -- example: [5051.0, 26137.0, 4004.0]
  "id_votesmart" REAL, -- example: [27018.0, 27122.0, 26888.0]
  "id_lis" TEXT, -- example: ['S307', 'S275', 'S308']
  "id_ballotpedia" TEXT, -- example: ['Sherrod Brown', 'Maria Cantwell', 'Ben Cardin']
  "id_opensecrets" TEXT, -- example: ['N00003535', 'N00007836', 'N00001955']
  "id_fec_0" TEXT, -- example: ['H2OH13033', 'S8WA00194', 'H6MD03177']
  "id_fec_1" TEXT, -- example: ['S6OH00163', 'H2WA01054', 'S6MD03177']
  "id_fec_2" TEXT -- example: ['S4TN00096', 'S0NV00237', 'S8ND00120']
)

CREATE TABLE "skills_job_dim" (
"job_id" INTEGER, -- example: [310991, 471015, 159380]
  "skill_id" INTEGER -- example: [1, 0, 169]
)

CREATE TABLE "job_postings_fact" (
"job_id" INTEGER, -- example: [1422666, 399976, 1541644]
  "company_id" INTEGER, -- example: [58904, 939, 1072]
  "job_title_short" TEXT, -- example: ['Data Analyst', 'Senior Data Engineer', 'Senior Data Analyst']
  "job_title" TEXT, -- example: ['JUNIOR IT DATA ANALYST (DURBAN)', 'Trainee, L1 integration and data analytics', 'Senior Data Engineer H/F']
  "job_location" TEXT, -- example: ['Durban, South Africa', 'Oulu, Finland', 'Lyon, France']
  "job_via" TEXT, -- example: ['via Pnet', 'via Nokia - Talentify', 'via LinkedIn']
  "job_schedule_type" TEXT, -- example: ['Full-time', 'Contractor', 'Internship']
  "job_work_from_home" INTEGER, -- example: [0, 1]
  "search_location" TEXT, -- example: ['South Africa', 'Finland', 'France']
  "job_posted_date" TEXT, -- example: ['2023-01-09 12:31:15', '2023-03-02 08:32:37', '2023-02-15 10:36:53']
  "job_no_degree_mention" INTEGER, -- example: [1, 0]
  "job_health_insurance" INTEGER, -- example: [0, 1]
  "job_country" TEXT, -- example: ['South Africa', 'Finland', 'France']
  "salary_rate" TEXT, -- example: ['year', 'hour', 'month']
  "salary_year_avg" REAL, -- example: [300000.0, 120000.0, 99150.0]
  "salary_hour_avg" REAL -- example: [77.5, 27.979999542236328, 27.5]
)

CREATE TABLE "alien_data" (
"id" INTEGER, -- example: [1, 2, 3]
  "first_name" TEXT, -- example: ['tyrus', 'ealasaid', 'violette']
  "last_name" TEXT, -- example: ['wrey', 'st louis', 'sawood']
  "email" TEXT, -- example: ['twrey0@sakura.ne.jp', 'estlouis1@amazon.co.uk', 'vsawood2@yolasite.com']
  "gender" TEXT, -- example: ['non-binary', 'female', 'male']
  "type" TEXT, -- example: ['reptile', 'flatwoods', 'nordic']
  "birth_year" INTEGER, -- example: [1717, 1673, 1675]
  "age" INTEGER, -- example: [307, 351, 349]
  "favorite_food" TEXT, -- example: ['white-faced tree rat', 'lizard, goanna', 'indian red admiral']
  "feeding_frequency" TEXT, -- example: ['weekly', 'seldom', 'often']
  "aggressive" INTEGER, -- example: [1, 0]
  "occupation" TEXT, -- example: ['senior cost accountant', 'senior sales associate', 'registered nurse']
  "current_location" TEXT, -- example: ['cincinnati', 'bethesda', 'oakland']
  "state" TEXT, -- example: ['ohio', 'maryland', 'california']
  "us_region" TEXT, -- example: ['great lakes', 'mideast', 'far west']
  "country" TEXT -- example: ['united states']
)

CREATE TABLE "cities_countries" (
"country_id" INTEGER, -- example: [1, 2, 4]
  "country_name" TEXT, -- example: ['afghanistan', 'albania', 'algeria']
  "country_code_2" TEXT, -- example: ['af', 'al', 'dz']
  "country_code_3" TEXT, -- example: ['afg', 'alb', 'dza']
  "region" TEXT, -- example: ['asia', 'europe', 'africa']
  "sub_region" TEXT, -- example: ['southern asia', 'southern europe', 'northern africa']
  "intermediate_region" TEXT, -- example: ['middle africa', 'caribbean', 'south america']
  "created_on" TEXT -- example: ['2024-07-18']
)

CREATE TABLE "legislation_date_dim" (
"date" TEXT, -- example: ['1917-01-01', '1917-01-02', '1917-01-03']
  "month_name" TEXT, -- example: ['January', 'February', 'March']
  "day_of_month" INTEGER -- example: [1, 2, 3]
)

CREATE TABLE "cities" (
"city_id" INTEGER, -- example: [1, 2, 3]
  "city_name" TEXT, -- example: ['tokyo', 'jakarta', 'delhi']
  "latitude" REAL, -- example: [139.6922, 106.8275, 77.23]
  "longitude" REAL, -- example: [35.6897, -6.175, 28.61]
  "country_code_2" TEXT, -- example: ['jp', 'id', 'in']
  "capital" INTEGER, -- example: [1, 0]
  "population" REAL, -- example: [37732000.0, 33756000.0, 32226000.0]
  "insert_date" TEXT -- example: ['2022-01-12', '2021-08-22', '2021-03-11']
)

CREATE TABLE "aliens_location" (
"loc_id" INTEGER, -- example: [1, 2, 3]
  "current_location" TEXT, -- example: ['Cincinnati', 'Bethesda', 'Oakland']
  "state" TEXT, -- example: ['Ohio', 'Maryland', 'California']
  "country" TEXT, -- example: ['United States']
  "occupation" TEXT -- example: ['Senior Cost Accountant', 'Senior Sales Associate', 'Registered Nurse']
)

CREATE TABLE "aliens" (
"id" INTEGER, -- example: [1, 2, 3]
  "first_name" TEXT, -- example: ['Tyrus', 'Ealasaid', 'Violette']
  "last_name" TEXT, -- example: ['Wrey', 'St Louis', 'Sawood']
  "email" TEXT, -- example: ['twrey0@sakura.ne.jp', 'estlouis1@amazon.co.uk', 'vsawood2@yolasite.com']
  "gender" TEXT, -- example: ['Agender', 'Female', 'Male']
  "type" TEXT, -- example: ['Reptile', 'Flatwoods', 'Nordic']
  "birth_year" INTEGER -- example: [1717, 1673, 1675]
)

CREATE TABLE "cities_languages" (
"language_id" INTEGER, -- example: [1, 2, 3]
  "language" TEXT, -- example: ['pashto', 'persian', 'uzbek']
  "country_code_2" TEXT -- example: ['af', 'al', 'dz']
)

CREATE TABLE "job_company" (
"company_id" INTEGER, -- example: [195094, 211890, 322965]
  "name" TEXT, -- example: ['Kaderabotim.bg', 'acalerate', 'Group S']
  "link" TEXT, -- example: ['http://www.bitplane.com/', 'http://www.face2face.eu/', 'http://www.electriccapital.com/']
  "link_google" TEXT, -- example (truncated): ['https://www.google.com/search?sca_esv=590391945&gl=us&hl=en&q=Kaderabotim.bg&sa=X&ved=0ahUKEwiOhvWe5...', 'https://www.google.com/search?gl=us&hl=en&q=acalerate&sa=X&ved=0ahUKEwjarf7iq7iAAxVjF1kFHS6SDlcQmJAC...', 'https://www.google.com/search?sca_esv=562289703&hl=en&gl=us&q=Group+S&sa=X&ved=0ahUKEwiKubbQ6Y2BAxXw...']
  "thumbnail" TEXT -- example (truncated): ['https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQucyCV4n5JSjRToCftcFslOpQtGA5sWqzQ5n9qDTc&s', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQCm6EZgIRlEbYcGXoDJebGoHBEYxzYRVLeizWvrxdzl450...', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSkDpU8eq0Ij29POYRRz6dtKvhrrefo11UlVhKY&s=0']
)

