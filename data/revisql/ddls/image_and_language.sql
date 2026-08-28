CREATE TABLE ATT_CLASSES (
    `ATT_CLASS_ID` integer, -- ATTRIBUTE CLASS ID. the unique attribute class ids.
    `ATT_CLASS` text, -- ATTRIBUTE CLASS. the corresponding classes for attributes.
    primary key (ATT_CLASS_ID)
);

CREATE TABLE IMG_OBJ_ATT (
    `IMG_ID` integer, -- IMAGE ID. id number of each image.
    `ATT_CLASS_ID` integer, -- ATTRIBUTE CLASS ID. attribute class number for image. if one IMG_ID has many ATT_CLASS_ID, it refers to that this image has multiple attributes.
    `OBJ_SAMPLE_ID` integer, -- OBJECT SAMPLE ID. object sample id. â¢ if one IMG_ID has many OBJ_SAMPLE_ID, it refers to that this image has multiple objects. â¢ if one ATT_CLASS_ID has many OBJ_SAMPLE_ID, it refers to that this attribute is composed of multiple objects.
    primary key (IMG_ID, ATT_CLASS_ID, OBJ_SAMPLE_ID),
    foreign key (ATT_CLASS_ID) references ATT_CLASSES (ATT_CLASS_ID),
    foreign key (IMG_ID, OBJ_SAMPLE_ID) references IMG_OBJ (IMG_ID, OBJ_SAMPLE_ID)
);

CREATE TABLE OBJ_CLASSES (
    `OBJ_CLASS_ID` integer, -- OBJECT CLASS ID. unique id number identifying object classes.
    `OBJ_CLASS` text, -- OBJECT CLASS. the explanation about object classes.
    primary key (OBJ_CLASS_ID)
);

CREATE TABLE IMG_OBJ (
    `IMG_ID` integer, -- IMAGE ID. the id representing images.
    `OBJ_SAMPLE_ID` integer, -- OBJECT SAMPLE ID. the id of the object sample.
    `OBJ_CLASS_ID` integer, -- OBJECT CLASS ID. the id indicating class of the objects.
    `X` integer, -- x coordinate.
    `Y` integer, -- y coordinate.
    `W` integer, -- width of the bounding box of the object.
    `H` integer, -- height of the bounding box of the object. bounding box of the object: (x, y, W, H).
    primary key (IMG_ID, OBJ_SAMPLE_ID),
    foreign key (OBJ_CLASS_ID) references OBJ_CLASSES (OBJ_CLASS_ID)
);

CREATE TABLE IMG_REL (
    `IMG_ID` integer, -- IMAGE ID. the image id.
    `PRED_CLASS_ID` integer, -- PREDICTION CLASS ID. the prediction relationship class between objects. if OBJ1_SAMPLE_ID = OBJ2_SAMPLE_ID, this relationship is the self-relation.
    `OBJ1_SAMPLE_ID` integer, -- OBJECT1 SAMPLE ID. the sample id of the first object.
    `OBJ2_SAMPLE_ID` integer, -- OBJECT2 SAMPLE ID. the sample id of the second object. if (OBJ1_SAMPLE_ID, OBJ2_SAMPLE_ID) has multiple PRED_CLASS_ID, it means these two objects have multiple relations.
    primary key (IMG_ID, PRED_CLASS_ID, OBJ1_SAMPLE_ID, OBJ2_SAMPLE_ID),
    foreign key (PRED_CLASS_ID) references PRED_CLASSES (PRED_CLASS_ID),
    foreign key (IMG_ID, OBJ1_SAMPLE_ID) references IMG_OBJ (IMG_ID, OBJ_SAMPLE_ID),
    foreign key (IMG_ID, OBJ2_SAMPLE_ID) references IMG_OBJ (IMG_ID, OBJ_SAMPLE_ID)
);

CREATE TABLE PRED_CLASSES (
    `PRED_CLASS_ID` integer, -- PREDICTION CLASS ID. the unique prediction id for the class.
    `PRED_CLASS` text, -- PREDICTION CLASS. the caption for the prediction class id.
    primary key (PRED_CLASS_ID)
);

