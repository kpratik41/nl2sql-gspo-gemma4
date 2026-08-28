CREATE TABLE projects (
    `projectid` text, -- project id. project's unique identifier.
    `teacher_acctid` text, -- teacher's unique identifier (teacher that created a project).
    `schoolid` text, -- school's identifier (school where teacher works).
    `school_ncesid` text, -- school national center for education statistics id. Public National Center for Ed Statistics id.
    `school_latitude` real, -- school latitude. latitude of school.
    `school_longitude` real, -- school longitude. longitude of school.
    `school_city` text, -- school city. city where the school is located.
    `school_state` text, -- school state. state where the school is located.
    `school_zip` integer, -- school zip. zipcode of school.
    `school_metro` text, -- school metro. metro where the school is located.
    `school_district` text, -- school district. district where the school is located.
    `school_county` text, -- school county. country where the school is located.
    `school_charter` text, -- school charter. whether it is a public charter school or not (no private schools in the dataset).
    `school_magnet` text, -- school magnet. whether a public magnet school or not.
    `school_year_round` text, -- school year round. whether it is a public year round school or not.
    `school_nlns` text, -- school Nonleafy Normal Stature. whether it is a public Nonleafy Normal Stature school or not.
    `school_kipp` text, -- school Knowledge Is Power Program. whether it is a public Knowledge Power Program school or not.
    `school_charter_ready_promise` text, -- school charter ready promise. whether it is a public ready promise school or not.
    `teacher_prefix` text, -- teacher prefix. teacher's gender. if the value is equal to "Dr. ", it means this teacher acquired P.h.D or doctor degree.
    `teacher_teach_for_america` text, -- teachers teach for america. Teach for America or not.
    `teacher_ny_teaching_fellow` text, -- teacher_new_york_teaching_fellow. New York teaching fellow or not. • t: true. • f: false. if true: it means that this teacher is a New York teacher.
    `primary_focus_subject` text, -- primary focus subject. main subject for which project materials are intended.
    `primary_focus_area` text, -- primary focus area. main subject area for which project materials are intended.
    `secondary_focus_subject` text, -- secondary focus subject. secondary subject.
    `secondary_focus_area` text, -- secondary focus area. secondary subject area.
    `resource_type` text, -- resource type. main type of resources requested by a project.
    `poverty_level` text, -- poverty level. school's poverty level. highest: 65%+ free of reduced lunch; high: 40-64%; moderate: 10-39%low: 0-9%.
    `grade_level` text, -- grade level. grade level for which project materials are intended.
    `fulfillment_labor_materials` real, -- fulfillment labor materials. cost of fulfillment. higher means higher cost, or need more labors.
    `total_price_excluding_optional_support` real, -- total price excluding optional support. project cost excluding optional tip that donors give to DonorsChoose.org while funding a project.
    `total_price_including_optional_support` real, -- total price including optional support. project cost including optional tip that donors give to DonorsChoose.org while funding a project. cost of optional tip = total_price_including_optional_support -total_price_excluding_optional_support.
    `students_reached` integer, -- students reached. number of students impacted by a project (if funded).
    `eligible_double_your_impact_match` text, -- eligible double your impact match. project was eligible for a 50% off offer by a corporate partner (logo appears on a project, like Starbucks or Disney).
    `eligible_almost_home_match` text, -- eligible almost home match. project was eligible for a $100 boost offer by a corporate partner.
    `date_posted` date, -- date posted. data a project went live on the site.
    primary key (projectid)
);

CREATE TABLE essays (
    `projectid` text, -- project id. unique project identifier.
    `teacher_acctid` text, -- teacher acctid. teacher id that created a project.
    `title` text, -- title of the project.
    `short_description` text, -- short description. short description of a project.
    `need_statement` text, -- need statement. need statement of a project.
    `essay` text, -- complete project essay.

);

CREATE TABLE resources (
    `resourceid` text, -- resource id. unique resource id.
    `projectid` text, -- project id. project id that requested resources for a classroom.
    `vendorid` integer, -- vendor id. vendor id that supplies resources to a project.
    `vendor_name` text, -- vendor name.
    `project_resource_type` text, -- project resource type. type of resource.
    `item_name` text, -- item name. resource name.
    `item_number` text, -- item number. resource item identifier.
    `item_unit_price` real, -- item unit price. unit price of the resource.
    `item_quantity` integer, -- item quantity. number of a specific item requested by a teacher.
    primary key (resourceid),
    foreign key (projectid) references projects(projectid)
);

CREATE TABLE donations (
    `donationid` text, -- donation id. unique donation identifier.
    `projectid` text, -- project id. project identifier that received the donation.
    `donor_acctid` text, -- donor accid. donor that made a donation.
    `donor_city` text, -- donor city. donation city.
    `donor_state` text, -- donor state. donation state.
    `donor_zip` text, -- donor zip. donation zip code.
    `is_teacher_acct` text, -- is teacher acct. whether donor is also a teacher. • f: false, it means this donor is not a teacher. • t: true, it means this donor is a teacher.
    `donation_timestamp` datetime, -- donation timestamp. the time of donation.
    `donation_to_project` real, -- donation to project. amount to project, excluding optional support (tip).
    `donation_optional_support` real, -- donation optional support. amount of optional support.
    `donation_total` real, -- donation total. donated amount. donated amount = donation_to_project + donation_optional_support.
    `dollar_amount` text, -- dollar amount. donated amount in US dollars.
    `donation_included_optional_support` text, -- donation included optional support. whether optional support (tip) was included for DonorsChoose.org. • f: false, optional support (tip) was not included for DonorsChoose.org. • t: true, optional support (tip) was included for DonorsChoose.org.
    `payment_method` text, -- payment method. what card/payment option was used.
    `payment_included_acct_credit` text, -- payment included acct credit. whether a portion of a donation used account credits redemption. • f: false, a portion of a donation didn't use account credits redemption. • t: true, a portion of a donation used account credits redemption.
    `payment_included_campaign_gift_card` text, -- payment included campaign gift card. whether a portion of a donation included corporate sponsored gift card. • f: false, a portion of a donation included corporate sponsored gift card. • t: true, a portion of a donation included corporate sponsored gift card.
    `payment_included_web_purchased_gift_card` text, -- payment included web purchased gift card. whether a portion of a donation included citizen purchased gift card (ex: friend buy a gift card for you). • f: false. • t: true.
    `` text, -- No description.
    `payment_was_promo_matched` text, -- payment was promo matched. whether a donation was matched 1-1 with corporate funds. • f: false, a donation was not matched 1-1 with corporate funds. • t: true, a donation was matched 1-1 with corporate funds.
    `via_giving_page` text, -- via giving page. donation given via a giving / campaign page (example: Mustaches for Kids).
    `for_honoree` text, -- for honoree. donation made for an honoree. • f: false, this donation is not made for an honoree. • t: true, this donation is made for an honoree.
    `donation_message` text, -- donation message. donation comment/message.
    primary key (donationid),
    foreign key (projectid) references projects(projectid)
);

