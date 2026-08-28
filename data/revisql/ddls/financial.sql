CREATE TABLE loan (
    `loan_id` integer, -- the id number identifying the loan data.
    `account_id` integer, -- the id number identifying the account.
    `date` date, -- the date when the loan is approved.
    `amount` integer, -- approved amount. unit：US dollar.
    `duration` integer, -- loan duration. unit：month.
    `payments` real, -- monthly payments. unit：month.
    `status` text, -- repayment status. 'A' stands for contract finished, no problems; 'B' stands for contract finished, loan not paid; 'C' stands for running contract, OK so far; 'D' stands for running contract, client in debt.
    primary key (loan_id),
    foreign key (account_id) references account (account_id)
);

CREATE TABLE client (
    `client_id` integer, -- the unique number.
    `gender` text, -- F：female. M：male.
    `birth_date` date, -- birth date.
    `district_id` integer, -- location of branch.
    primary key (client_id),
    foreign key (district_id) references district (district_id)
);

CREATE TABLE district (
    `district_id` integer , -- location of branch.
    `A2` text, -- district_name.
    `A3` text, -- region.
    `A4` text, -- number of inhabitants.
    `A5` text, -- no. of municipalities with inhabitants < 499. municipality < district < region.
    `A6` text, -- no. of municipalities with inhabitants 500-1999. municipality < district < region.
    `A7` text, -- no. of municipalities with inhabitants 2000-9999. municipality < district < region.
    `A8` integer, -- no. of municipalities with inhabitants > 10000. municipality < district < region.
    `A9` integer, -- not useful.
    `A10` real, -- ratio of urban inhabitants.
    `A11` integer, -- average salary.
    `A12` real, -- unemployment rate 1995.
    `A13` real, -- unemployment rate 1996.
    `A14` integer, -- no. of entrepreneurs per 1000 inhabitants.
    `A15` integer, -- no. of committed crimes 1995.
    `A16` integer, -- no. of committed crimes 1996.
    primary key (district_id)
);

CREATE TABLE trans (
    `trans_id` integer, -- transaction id.
    `account_id` integer, -- No description.
    `date` date, -- date of transaction.
    `type` text, -- +/- transaction. "PRIJEM" stands for credit. "VYDAJ" stands for withdrawal.
    `operation` text, -- mode of transaction. "VYBER KARTOU": credit card withdrawal. "VKLAD": credit in cash. "PREVOD Z UCTU" :collection from another bank. "VYBER": withdrawal in cash. "PREVOD NA UCET": remittance to another bank.
    `amount` integer, -- amount of money. Unit：USD.
    `balance` integer, -- balance after transaction. Unit：USD.
    `k_symbol` text, -- characterization of the transaction. "POJISTNE": stands for insurrance payment. "SLUZBY": stands for payment for statement. "UROK": stands for interest credited. "SANKC. UROK": sanction interest if negative balance. "SIPO": stands for household. "DUCHOD": stands for old-age pension. "UVER": stands for loan payment.
    `bank` text, -- bank of the partner. each bank has unique two-letter code.
    `account` integer, -- account of the partner.
    primary key (trans_id),
    foreign key (account_id) references account (account_id)
);

CREATE TABLE account (
    `account_id` integer, -- account id. the id of the account.
    `district_id` integer , -- location of branch.
    `frequency` text, -- frequency of the acount.
    `date` date, -- the creation date of the account. in the form YYMMDD.
    primary key (account_id),
    foreign key (district_id) references district (district_id)
);

CREATE TABLE card (
    `card_id` integer, -- credit card id. id number of credit card.
    `disp_id` integer, -- disposition id.
    `type` text, -- type of credit card. "junior": junior class of credit card; "classic": standard class of credit card; "gold": high-level credit card.
    `issued` date, -- the date when the credit card issued. in the form YYMMDD.
    primary key (card_id),
    foreign key (disp_id) references disp (disp_id)
);

CREATE TABLE order (
    `order_id` integer, -- identifying the unique order.
    `account_id` integer, -- id number of account.
    `bank_to` text, -- bank of the recipient.
    `account_to` integer, -- account of the recipient. each bank has unique two-letter code.
    `amount` real, -- debited amount.
    `k_symbol` text, -- characterization of the payment. purpose of the payment. "POJISTNE" stands for insurance payment. "SIPO" stands for household payment. "LEASING" stands for leasing. "UVER" stands for loan payment.
    primary key (order_id),
    foreign key (account_id) references account (account_id)
);

CREATE TABLE disp (
    `disp_id` integer, -- disposition id. unique number of identifying this row of record.
    `client_id` integer, -- id number of client.
    `account_id` integer, -- id number of account.
    `type` text, -- type of disposition. "OWNER" : "USER" : "DISPONENT". the account can only have the right to issue permanent orders or apply for loans.
    primary key (disp_id),
    foreign key (account_id) references account (account_id),
    foreign key (client_id) references client (client_id)
);

