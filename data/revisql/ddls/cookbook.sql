CREATE TABLE Quantity (
    `quantity_id` integer, -- quantity id. the unique identifier for the quantity.
    `recipe_id` integer, -- recipe id. the id of the recipe.
    `ingredient_id` integer, -- ingredient id. the id of the ingredient.
    `max_qty` real, -- maximum quantity. the max quantity of the ingredient.
    `min_qty` real, -- minimum quantity. the min quantity of the ingredient. If max_qty equals to min_qty, it means that the ingredient must be rationed.
    `unit` text, -- the unit of the ingredient.
    `preparation` text, -- 'No data' means the ingredient doesn't need preprocessing.
    `optional` text, -- whether the ingredient is optional.
    primary key (quantity_id),
    foreign key (recipe_id) references Recipe(recipe_id),
    foreign key (ingredient_id) references Ingredient(ingredient_id),
    foreign key (recipe_id) references Nutrition(recipe_id)
);

CREATE TABLE Nutrition (
    `recipe_id` integer, -- recipe id. the id of the recipe.
    `protein` real, -- the protein content in the recipe.
    `carbo` real, -- the carbo content in the recipe.
    `alcohol` real, -- the alcohol content in the recipe.
    `total_fat` real, -- total fat. the total fat content in the recipe. higher --> higher possibility to gain weight.
    `sat_fat` real, -- saturated fat. the saturated fat content in the recipe. unsaturated fat = total_fat - saturated fat.
    `cholestrl` real, -- cholesterol. the cholesterol content in the recipe.
    `sodium` real, -- the sodium content in the recipe. Salt/Sodium-Free - Less than 5 mg of sodium per serving. Very Low Sodium - 35 mg of sodium or less per serving. Low Sodium -140 mg of sodium or less per serving. Reduced Sodium - At least 25% less sodium than the regular product. Light in Sodium or Lightly Salted - At least 50% less sodium than the regular product. No-Salt-Added or Unsalted - No salt is added during processing - but these products may not be salt/sodium-free unless stated.
    `iron` real, -- the iron content in the recipe. if iron > 20mg, it will lead to.  constipation.  feeling sick.  being sick.  stomach pain. question could mention any of functions listed before.
    `vitamin_c` real, -- vitamin c. the vitamin c content in the recipe. Vitamin C, also known as ascorbic acid, if the VC is higher, it will:.  helping to protect cells and keeping them healthy.  maintaining healthy skin, blood vessels, bones and cartilage.  helping with wound healing. question could mention any of functions listed before.
    `vitamin_a` real, -- vitamin a. the vitamin a content in the recipe. higher --> beneficial to.  helping your body's natural defense against illness and infection (the immune system) work properly.  helping vision in dim light.  keeping skin and the lining of some parts of the body, such as the nose, healthy. question could mention any of functions listed before.
    `fiber` real, -- the fiber a content in the recipe.
    `pcnt_cal_carb` real, -- percentage calculation carbo. percentage of carbo in total nutrient composition.
    `pcnt_cal_fat` real, -- percentage calculation fat. percentage of fat in total nutrient composition.
    `pcnt_cal_prot` real, -- percentage calculation protein. percentage of protein in total nutrient composition.
    `calories` real, -- the calories of the recipe.
    primary key (recipe_id),
    foreign key (recipe_id) references Recipe(recipe_id)
);

CREATE TABLE Recipe (
    `recipe_id` integer, -- recipe id. the unique identifier for the recipe.
    `title` text, -- the title of the recipe.
    `subtitle` text, -- the subtitle of the recipe.
    `servings` integer, -- the number of people the recipe can serve.
    `yield_unit` text, -- yield unit. the unit of the yield for the recipe.
    `prep_min` integer, -- preparation minute. the preparation minute of the recipe.
    `cook_min` integer, -- cooked minute. the cooked minute of the recipe.
    `stnd_min` integer, -- stand minute. the stand minute of the price. The stand minute stands for the time to take the dish away from the heat source. the total time of the recipe = prep_min + cook_min + stnd_min.
    `source` text, -- the source of the recipe.
    `intro` text, -- introduction. the introduction of the recipe.
    `directions` text, -- the directions of the recipe.
    primary key (recipe_id)
);

CREATE TABLE Ingredient (
    `ingredient_id` integer, -- ingredient id. the unique identifier for the ingredient.
    `category` text, -- the category of the ingredient.
    `name` text, -- the name of the ingredient.
    `plural` text, -- the plural suffix of the ingredient.
    primary key (ingredient_id)
);

