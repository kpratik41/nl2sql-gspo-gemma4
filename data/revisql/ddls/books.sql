CREATE TABLE customer (
    `customer_id` integer, -- customer id. the unique identifier of the customer.
    `first_name` text, -- first name. the first name of the customer.
    `last_name` text, -- last name. the last name of the customer. A person's full name is the first name, middle name (if applicable), and last name.
    `email` text, -- the email of the customer.
    primary key (customer_id)
);

CREATE TABLE cust_order (
    `order_id` integer , -- order id. the unique identifier of the customer order.
    `order_date` datetime, -- order date. the date of the order.
    `customer_id` integer, -- customer id. the id of the customer. Maps to customer(customer_Id). The number of orders ordered by the customer = the show-up times of the relative customer id in the table.
    `shipping_method_id` integer, -- shipping method id. the id of the shipping method. Maps to shipping_method(method_id).
    `dest_address_id` integer, -- destination address id. the id of the destination address. Maps to address(address_id).
    primary key (order_id)
);

CREATE TABLE shipping_method (
    `method_id` integer, -- method id. the unique identifier of the method.
    `method_name` text, -- method name. the method name.
    `cost` real, -- the cost of the shipping method. The main difference between the various shipping methods, such as standard, priority, express, and international, is the speed at which the item is delivered. Standard shipping is the slowest and least expensive option, while express and priority shipping are faster and more expensive. International shipping is for items that are being shipped to a destination outside of the country where they originated.
    primary key (method_id)
);

CREATE TABLE book_author (
    `book_id` integer, -- book id. the id of the book. Maps to book(book_id).
    `author_id` integer, -- author id. the id of the author. Maps to author(author_id). Books with the same author id are written by the same author.
    primary key (book_id, author_id),
    foreign key (author_id) references author(author_id),
    foreign key (book_id) references book(book_id)
);

CREATE TABLE book_language (
    `language_id` integer, -- language id. the unique identifier of the language.
    `language_code` text, -- language code. the language code. A language code is a unique identifier for a specific language. It is typically used to identify a language in computing and other technical contexts. Some common language codes include "en" for English, "fr" for French, and "es" for Spanish. The specific codes used for each language can vary depending on the context and the system in which they are being used.
    `language_name` text, -- language name. the language name.
    primary key (language_id)
);

CREATE TABLE order_status (
    `status_id` integer, -- status id. the unique identifier of the order status.
    `status_value` text, -- status value. the status value. The order statuses include order received, pending delivery, delivery in progress, delivered, canceled, and returned.
    primary key (status_id)
);

CREATE TABLE address_status (
    `status_id` integer, -- status id. the unique identifier of the status.
    `address_status` text, -- address status. the status of the address.  active: the address is still in use.  inactive: the address is not in use anymore.
    primary key (status_id)
);

CREATE TABLE author (
    `author_id` integer, -- author id. the unique identifier of the author.
    `author_name` text, -- author name. the name of the author.
    primary key (author_id)
);

CREATE TABLE publisher (
    `publisher_id` integer, -- publisher id. the unique identifier of the publisher.
    `publisher_name` text, -- publisher name. the name of the publisher.
    primary key (publisher_id)
);

CREATE TABLE country (
    `country_id` integer, -- country id. the unique identifier of the country.
    `country_name` text, -- country name. the country name.
    primary key (country_id)
);

CREATE TABLE order_line (
    `line_id` integer , -- line id. the unique identifier of the order line.
    `order_id` integer, -- order id. the id of the order. Maps to cust_order(order_id).
    `book_id` integer, -- book id. the id of the book. Maps to book(book_id).
    `price` real, -- the price of the order. Even though the customer ordered the book with the same book id, the price could be different. The price of the order may be influenced by the shipping method, seasonal discount, and the number of books the customer ordered.
    primary key (line_id)
);

CREATE TABLE customer_address (
    `customer_id` integer, -- customer id. the id of the customer. Maps to customer(customer_id).
    `address_id` integer, -- address id. the id of the address. Maps to address(address_id).
    `status_id` integer, -- status id. the id of the address status. A customer may have several addresses. If a customer has several addresses, the address that the status_id = 1 is the customer's current address that is in use. The other addresses with 2 as status_id is abandoned addresses.
    primary key (customer_id, address_id),
    foreign key (address_id) references address(address_id),
    foreign key (customer_id) references customer(customer_id)
);

CREATE TABLE address (
    `address_id` integer, -- address id. the unique identifier of the address.
    `street_number` text, -- street number. the street number of the address. The street number is typically used to identify a specific building or residence on a street, with higher numbers generally indicating a location further down the street from the starting point. For example, if a street starts at number 1 and goes up to number 100, a building with the number 50 would be closer to the starting point than a building with the number 75.
    `street_name` text, -- street name. the street name.
    `city` text, -- the city where the address belongs.
    `country_id` integer, -- country id. the id of the country where the address belongs. Maps to the country (country id). The full address of the customer is 'No.street_number street_name, city, country'.
    primary key (address_id),
    foreign key (country_id) references country(country_id)
);

CREATE TABLE order_history (
    `history_id` integer , -- history id. the unique identifier of the order history.
    `order_id` integer, -- the id of the order. Maps to cust_order(order_id).
    `status_id` integer, -- status id. the id of the order. Maps to order_status(status_id). The order statuses include order received, pending delivery, delivery in progress, delivered, canceled, and returned.
    `status_date` datetime, -- status date. the date of the status updated.
    primary key (history_id)
);

CREATE TABLE book (
    `book_id` integer, -- book id. the unique identifier of the book.
    `title` text, -- the title of the book.
    `isbn13` text, -- the International Standard Book Number of the book. An ISBN is a unique 13-digit number assigned to each book to identify it internationally. The ISBN13 of a book is the specific version of the ISBN that uses 13 digits. It is typically displayed on the back cover of a book, along with the barcode and other information.
    `language_id` integer, -- language id. the id of the book language. Maps to book_language (language_id).
    `num_pages` integer, -- number pages. the number of the pages.
    `publication_date` date, -- publication date. the publication date of the book. The publication date of a book can provide information about when the book was released to the public. This can be useful for understanding the context in which the book was written, as well as for determining how current or outdated the information in the book might be. Additionally, the publication date can provide insight into the age of the book and its potential value as a collectible.
    `publisher_id` integer, -- the id of the publisher. Maps to publisher (publisher_id).
    primary key (book_id),
    foreign key (language_id) references book_language(language_id),
    foreign key (publisher_id) references publisher(publisher_id)
);

