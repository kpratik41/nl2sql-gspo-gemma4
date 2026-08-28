CREATE TABLE paper (
    `paper_id` text, -- paper id. unique string ID of the paper.
    `class_label` text, -- class label.
    primary key (paper_id)
);

CREATE TABLE cites (
    `cited_paper_id` text, -- cited_paper_id is the ID of the paper being cited.
    `citing_paper_id` text, -- citing_paper_id stands for the paper which contains the citation.
    primary key (cited_paper_id, citing_paper_id)
);

CREATE TABLE content (
    `paper_id` text, -- paper id. unique string ID of the paper.
    `word_cited_id` text, -- word cited id. rtype. whether each word in the vocabulary is present (indicated by 1) or absent (indicated by 0) in the paper.
    primary key (paper_id, word_cited_id),
    foreign key (paper_id) references paper(paper_id)
);

