CREATE TABLE bond (
    `bond_id` text, -- unique id representing bonds. TRxxx_A1_A2:. TRXXX refers to which molecule. A1 and A2 refers to which atom.
    `molecule_id` text, -- identifying the molecule in which the bond appears.
    `bond_type` text, -- type of the bond. -: single bond. '=': double bond. '#': triple bond.
    PRIMARY KEY (bond_id),
    FOREIGN KEY (molecule_id) REFERENCES molecule (molecule_id)
);

CREATE TABLE molecule (
    `molecule_id` text, -- molecule id. unique id of molecule. "+" --> this molecule / compound is carcinogenic. '-' this molecule is not / compound carcinogenic.
    `label` text, -- whether this molecule is carcinogenic or not.
    PRIMARY KEY (molecule_id)
);

CREATE TABLE atom (
    `atom_id` text, -- atom id. the unique id of atoms.
    `molecule_id` text, -- molecule id. identifying the molecule to which the atom belongs. TRXXX_i represents ith atom of molecule TRXXX.
    `element` text, -- the element of the toxicology.  cl: chlorine.  c: carbon.  h: hydrogen.  o: oxygen.  s: sulfur.  n: nitrogen.  p: phosphorus.  na: sodium.  br: bromine.  f: fluorine.  i: iodine.  sn: Tin.  pb: lead.  te: tellurium.  ca: Calcium.
    PRIMARY KEY (atom_id),
    FOREIGN KEY (molecule_id) REFERENCES molecule (molecule_id)
);

CREATE TABLE connected (
    `atom_id` text, -- atom id. id of the first atom.
    `atom_id2` text, -- atom id 2. id of the second atom.
    `bond_id` text, -- bond id. bond id representing bond between two atoms.
    PRIMARY KEY (atom_id,atom_id2),
    FOREIGN KEY (atom_id) REFERENCES atom (atom_id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (atom_id2) REFERENCES atom (atom_id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (bond_id) REFERENCES bond (bond_id) ON DELETE CASCADE ON UPDATE CASCADE
);

