CREATE TABLE gender (
    `id` integer, -- the unique identifier of the gender.
    `gender` text, -- the gender of the superhero.
    primary key (id)
);

CREATE TABLE superpower (
    `id` integer, -- the unique identifier of the superpower.
    `power_name` text, -- power name. the superpower name.
    primary key (id)
);

CREATE TABLE publisher (
    `id` integer, -- the unique identifier of the publisher.
    `publisher_name` text, -- the name of the publisher.
    primary key (id)
);

CREATE TABLE superhero (
    `id` integer, -- the unique identifier of the superhero.
    `superhero_name` text, -- superhero name. the name of the superhero.
    `full_name` text, -- full name. the full name of the superhero. The full name of a person typically consists of their given name, also known as their first name or personal name, and their surname, also known as their last name or family name. For example, if someone's given name is "John" and their surname is "Smith," their full name would be "John Smith.".
    `gender_id` integer, -- gender id. the id of the superhero's gender.
    `eye_colour_id` integer, -- eye colour id. the id of the superhero's eye color.
    `hair_colour_id` integer, -- hair colour id. the id of the superhero's hair color.
    `skin_colour_id` integer, -- skin colour id. the id of the superhero's skin color.
    `race_id` integer, -- race id. the id of the superhero's race.
    `publisher_id` integer, -- publisher id. the id of the publisher.
    `alignment_id` integer, -- alignment id. the id of the superhero's alignment.
    `height_cm` integer, -- height cm. the height of the superhero. The unit of height is centimeter. If the height_cm is NULL or 0, it means the height of the superhero is missing.
    `weight_kg` integer, -- weight kg. the weight of the superhero. The unit of weight is kilogram. If the weight_kg is NULL or 0, it means the weight of the superhero is missing.
    primary key (id),
    foreign key (alignment_id) references alignment(id),
    foreign key (eye_colour_id) references colour(id),
    foreign key (gender_id) references gender(id),
    foreign key (hair_colour_id) references colour(id),
    foreign key (publisher_id) references publisher(id),
    foreign key (race_id) references race(id),
    foreign key (skin_colour_id) references colour(id)
);

CREATE TABLE colour (
    `id` integer, -- the unique identifier of the color.
    `colour` text, -- the color of the superhero's skin/eye/hair/etc.
    primary key (id)
);

CREATE TABLE attribute (
    `id` integer, -- the unique identifier of the attribute.
    `attribute_name` text, -- attribute name. the attribute. A superhero's attribute is a characteristic or quality that defines who they are and what they are capable of. This could be a physical trait, such as superhuman strength or the ability to fly, or a personal trait, such as extraordinary intelligence or exceptional bravery.
    primary key (id)
);

CREATE TABLE hero_power (
    `hero_id` integer, -- hero id. the id of the hero. Maps to superhero(id).
    `power_id` integer, -- power id. the id of the power. Maps to superpower(id). In general, a superhero's attributes provide the foundation for their abilities and help to define who they are, while their powers are the specific abilities that they use to fight crime and protect others.
    foreign key (hero_id) references superhero(id),
    foreign key (power_id) references superpower(id)
);

CREATE TABLE race (
    `id` integer, -- the unique identifier of the race.
    `race` text, -- the race of the superhero. In the context of superheroes, a superhero's race would refer to the particular group of people that the superhero belongs to base on these physical characteristics.
    primary key (id)
);

CREATE TABLE alignment (
    `id` integer, -- the unique identifier of the alignment.
    `alignment` text, -- the alignment of the superhero. Alignment refers to a character's moral and ethical stance and can be used to describe the overall attitude or behavior of a superhero. Some common alignments for superheroes include:. Good: These superheroes are typically kind, selfless, and dedicated to protecting others and upholding justice. Examples of good alignments include Superman, Wonder Woman, and Spider-Man. Neutral: These superheroes may not always prioritize the greater good, but they are not necessarily evil either. They may act in their own self-interest or make decisions based on their own moral code. Examples of neutral alignments include the Hulk and Deadpool.  Bad: These superheroes are typically selfish, manipulative, and willing to harm others in pursuit of their own goals. Examples of evil alignments include Lex Luthor and the Joker.
    primary key (id)
);

CREATE TABLE hero_attribute (
    `hero_id` integer, -- hero id. the id of the hero. Maps to superhero(id).
    `attribute_id` integer, -- attribute id. the id of the attribute. Maps to attribute(id).
    `attribute_value` integer, -- attribute value. the attribute value. If a superhero has a higher attribute value on a particular attribute, it means that they are more skilled or powerful in that area compared to other superheroes. For example, if a superhero has a higher attribute value for strength, they may be able to lift heavier objects or deliver more powerful punches than other superheroes.
    foreign key (attribute_id) references attribute(id),
    foreign key (hero_id) references superhero(id)
);

