CREATE TABLE income (
    `income_id` text, -- income id. A unique identifier for each record of income.
    `date_received` text, -- date received. the date that the fund received.
    `amount` integer, -- amount of funds. the unit is dollar.
    `source` text, -- A value indicating where the funds come from such as dues, or the annual university allocation.
    `notes` text, -- A free-text value giving any needed details about the receipt of funds.
    `link_to_member` text, -- link to member.
    primary key (income_id),
    foreign key (link_to_member) references member(member_id)
);

CREATE TABLE budget (
    `budget_id` text, -- budget id. A unique identifier for the budget entry.
    `category` text, -- The area for which the amount is budgeted, such as, advertisement, food, parking.
    `spent` real, -- The total amount spent in the budgeted category for an event. the unit is dollar. This is summarized from the Expense table.
    `remaining` real, -- A value calculated as the amount budgeted minus the amount spent. the unit is dollar. If the remaining < 0, it means that the cost has exceeded the budget.
    `amount` integer, -- The amount budgeted for the specified category and event. the unit is dollar. some computation like: amount = spent + remaining.
    `event_status` text, -- event status. the status of the event. Closed / Open/ Planning. • Closed: It means that the event is closed. The spent and the remaining won't change anymore. • Open: It means that the event is already opened. The spent and the remaining will change with new expenses. • Planning: The event is not started yet but is planning. The spent and the remaining won't change at this stage.
    `link_to_event` text, -- link to event. The unique identifier of the event to which the budget line applies. References the Event table.
    primary key (budget_id),
    foreign key (link_to_event) references event(event_id)
);

CREATE TABLE zip_code (
    `zip_code` integer, -- zip code. The ZIP code itself. A five-digit number identifying a US post office.
    `type` text, -- The kind of ZIP code. ï¿½ Standard: the normal codes with which most people are familiar. ï¿½ PO Box: zip codes have post office boxes. ï¿½ Unique: zip codes that are assigned to individual organizations.
    `city` text, -- The city to which the ZIP pertains.
    `county` text, -- The county to which the ZIP pertains.
    `state` text, -- The name of the state to which the ZIP pertains.
    `short_state` text, -- short state. The abbreviation of the state to which the ZIP pertains.
    primary key (zip_code)
);

CREATE TABLE expense (
    `expense_id` text, -- expense id. unique id of income.
    `expense_description` text, -- expense description. A textual description of what the money was spend for.
    `expense_date` text, -- expense date. The date the expense was incurred. e.g. YYYY-MM-DD.
    `cost` real, -- The dollar amount of the expense. the unit is dollar.
    `approved` text, -- A true or false value indicating if the expense was approved. true/ false.
    `link_to_member` text, -- link to member. The member who incurred the expense.
    `link_to_budget` text, -- link to budget. The unique identifier of the record in the Budget table that indicates the expected total expenditure for a given category and event. References the Budget table.
    primary key (expense_id),
    foreign key (link_to_budget) references budget(budget_id),
    foreign key (link_to_member) references member(member_id)
);

CREATE TABLE member (
    `member_id` text, -- member id. unique id of member.
    `first_name` text, -- first name. member's first name.
    `last_name` text, -- last name. member's last name. full name is first_name + last_name. e.g. A member's first name is Angela and last name is Sanders. Thus, his/her full name is Angela Sanders.
    `email` text, -- member's email.
    `position` text, -- The position the member holds in the club.
    `t_shirt_size` text, -- The size of tee shirt that member wants when shirts are ordered. usually the student ordered t-shirt with lager size has bigger body shape.
    `phone` text, -- The best telephone at which to contact the member.
    `zip` integer, -- the zip code of the member's hometown.
    `link_to_major` text, -- link to major. The unique identifier of the major of the member. References the Major table.
    primary key (member_id),
    foreign key (link_to_major) references major(major_id),
    foreign key (zip) references zip_code(zip_code)
);

CREATE TABLE attendance (
    `link_to_event` text, -- link to event. The unique identifier of the event which was attended. References the Event table.
    `link_to_member` text, -- link to member. The unique identifier of the member who attended the event. References the Member table.
    primary key (link_to_event, link_to_member),
    foreign key (link_to_event) references event(event_id),
    foreign key (link_to_member) references member(member_id)
);

CREATE TABLE event (
    `event_id` text, -- event id. A unique identifier for the event.
    `event_name` text, -- event name.
    `event_date` text, -- event date. The date the event took place or is scheduled to take place. e.g. 2020-03-10T12:00:00.
    `type` text, -- The kind of event, such as game, social, election.
    `notes` text, -- A free text field for any notes about the event.
    `location` text, -- Address where the event was held or is to be held or the name of such a location.
    `status` text, -- One of three values indicating if the event is in planning, is opened, or is closed. Open/ Closed/ Planning.
    primary key (event_id)
);

CREATE TABLE major (
    `major_id` text, -- major id. A unique identifier for each major.
    `major_name` text, -- major name.
    `department` text, -- The name of the department that offers the major.
    `college` text, -- The name college that houses the department that offers the major.
    primary key (major_id)
);

