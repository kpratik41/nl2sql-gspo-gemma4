CREATE TABLE [Album]
(
    [AlbumId] INTEGER  NOT NULL, -- example: [1, 4, 2]
    [Title] NVARCHAR(160)  NOT NULL, -- example: ['For Those About To Rock We Salute You', 'Balls to the Wall', 'Restless and Wild']
    [ArtistId] INTEGER  NOT NULL, -- example: [1, 2, 3]
    CONSTRAINT [PK_Album] PRIMARY KEY  ([AlbumId]),
    FOREIGN KEY ([ArtistId]) REFERENCES [Artist] ([ArtistId]) 
		ON DELETE NO ACTION ON UPDATE NO ACTION
)

CREATE TABLE [Artist]
(
    [ArtistId] INTEGER  NOT NULL, -- example: [1, 2, 3]
    [Name] NVARCHAR(120), -- example: ['AC/DC', 'Accept', 'Aerosmith']
    CONSTRAINT [PK_Artist] PRIMARY KEY  ([ArtistId])
)

CREATE TABLE [Customer]
(
    [CustomerId] INTEGER  NOT NULL, -- example: [1, 3, 12]
    [FirstName] NVARCHAR(40)  NOT NULL, -- example: ['Luís', 'Leonie', 'François']
    [LastName] NVARCHAR(20)  NOT NULL, -- example: ['Gonçalves', 'Köhler', 'Tremblay']
    [Company] NVARCHAR(80), -- example: ['Embraer - Empresa Brasileira de Aeronáutica S.A.', 'JetBrains s.r.o.', 'Woodstock Discos']
    [Address] NVARCHAR(70), -- example: ['Av. Brigadeiro Faria Lima, 2170', 'Theodor-Heuss-Straße 34', '1498 rue Bélanger']
    [City] NVARCHAR(40), -- example: ['São José dos Campos', 'Stuttgart', 'Montréal']
    [State] NVARCHAR(40), -- example: ['SP', 'QC', 'RJ']
    [Country] NVARCHAR(40), -- example: ['Brazil', 'Germany', 'Canada']
    [PostalCode] NVARCHAR(10), -- example: ['12227-000', '70174', 'H2G 1A7']
    [Phone] NVARCHAR(24), -- example: ['+55 (12) 3923-5555', '+49 0711 2842222', '+1 (514) 721-4711']
    [Fax] NVARCHAR(24), -- example: ['+55 (12) 3923-5566', '+420 2 4172 5555', '+55 (11) 3033-4564']
    [Email] NVARCHAR(60)  NOT NULL, -- example: ['luisg@embraer.com.br', 'leonekohler@surfeu.de', 'ftremblay@gmail.com']
    [SupportRepId] INTEGER, -- example: [3, 4, 5]
    CONSTRAINT [PK_Customer] PRIMARY KEY  ([CustomerId]),
    FOREIGN KEY ([SupportRepId]) REFERENCES [Employee] ([EmployeeId]) 
		ON DELETE NO ACTION ON UPDATE NO ACTION
)

CREATE TABLE [Employee]
(
    [EmployeeId] INTEGER  NOT NULL, -- example: [1, 2, 6]
    [LastName] NVARCHAR(20)  NOT NULL, -- example: ['Adams', 'Edwards', 'Peacock']
    [FirstName] NVARCHAR(20)  NOT NULL, -- example: ['Andrew', 'Nancy', 'Jane']
    [Title] NVARCHAR(30), -- example: ['General Manager', 'Sales Manager', 'Sales Support Agent']
    [ReportsTo] INTEGER, -- example: [1, 2, 6]
    [BirthDate] DATETIME, -- example: ['1962-02-18 00:00:00', '1958-12-08 00:00:00', '1973-08-29 00:00:00']
    [HireDate] DATETIME, -- example: ['2002-08-14 00:00:00', '2002-05-01 00:00:00', '2002-04-01 00:00:00']
    [Address] NVARCHAR(70), -- example: ['11120 Jasper Ave NW', '825 8 Ave SW', '1111 6 Ave SW']
    [City] NVARCHAR(40), -- example: ['Edmonton', 'Calgary', 'Lethbridge']
    [State] NVARCHAR(40), -- example: ['AB']
    [Country] NVARCHAR(40), -- example: ['Canada']
    [PostalCode] NVARCHAR(10), -- example: ['T5K 2N1', 'T2P 2T3', 'T2P 5M5']
    [Phone] NVARCHAR(24), -- example: ['+1 (780) 428-9482', '+1 (403) 262-3443', '+1 (403) 263-4423']
    [Fax] NVARCHAR(24), -- example: ['+1 (780) 428-3457', '+1 (403) 262-3322', '+1 (403) 262-6712']
    [Email] NVARCHAR(60), -- example: ['andrew@chinookcorp.com', 'nancy@chinookcorp.com', 'jane@chinookcorp.com']
    CONSTRAINT [PK_Employee] PRIMARY KEY  ([EmployeeId]),
    FOREIGN KEY ([ReportsTo]) REFERENCES [Employee] ([EmployeeId]) 
		ON DELETE NO ACTION ON UPDATE NO ACTION
)

CREATE TABLE [Genre]
(
    [GenreId] INTEGER  NOT NULL, -- example: [1, 2, 3]
    [Name] NVARCHAR(120), -- example: ['Rock', 'Jazz', 'Metal']
    CONSTRAINT [PK_Genre] PRIMARY KEY  ([GenreId])
)

CREATE TABLE [Invoice]
(
    [InvoiceId] INTEGER  NOT NULL, -- example: [98, 121, 143]
    [CustomerId] INTEGER  NOT NULL, -- example: [1, 2, 3]
    [InvoiceDate] DATETIME  NOT NULL, -- example: ['2009-01-01 00:00:00', '2009-01-02 00:00:00', '2009-01-03 00:00:00']
    [BillingAddress] NVARCHAR(70), -- example: ['Theodor-Heuss-Straße 34', 'Ullevålsveien 14', 'Grétrystraat 63']
    [BillingCity] NVARCHAR(40), -- example: ['Stuttgart', 'Oslo', 'Brussels']
    [BillingState] NVARCHAR(40), -- example: ['AB', 'MA', 'Dublin']
    [BillingCountry] NVARCHAR(40), -- example: ['Germany', 'Norway', 'Belgium']
    [BillingPostalCode] NVARCHAR(10), -- example: ['70174', '0171', '1000']
    [Total] NUMERIC(10,2)  NOT NULL, -- example: [1.98, 3.96, 5.94]
    CONSTRAINT [PK_Invoice] PRIMARY KEY  ([InvoiceId]),
    FOREIGN KEY ([CustomerId]) REFERENCES [Customer] ([CustomerId]) 
		ON DELETE NO ACTION ON UPDATE NO ACTION
)

CREATE TABLE [InvoiceLine]
(
    [InvoiceLineId] INTEGER  NOT NULL, -- example: [579, 1, 1154]
    [InvoiceId] INTEGER  NOT NULL, -- example: [1, 2, 3]
    [TrackId] INTEGER  NOT NULL, -- example: [1, 2, 3]
    [UnitPrice] NUMERIC(10,2)  NOT NULL, -- example: [0.99, 1.99]
    [Quantity] INTEGER  NOT NULL, -- example: [1]
    CONSTRAINT [PK_InvoiceLine] PRIMARY KEY  ([InvoiceLineId]),
    FOREIGN KEY ([InvoiceId]) REFERENCES [Invoice] ([InvoiceId]) 
		ON DELETE NO ACTION ON UPDATE NO ACTION,
    FOREIGN KEY ([TrackId]) REFERENCES [Track] ([TrackId]) 
		ON DELETE NO ACTION ON UPDATE NO ACTION
)

CREATE TABLE [MediaType]
(
    [MediaTypeId] INTEGER  NOT NULL, -- example: [1, 2, 3]
    [Name] NVARCHAR(120), -- example: ['MPEG audio file', 'Protected AAC audio file', 'Protected MPEG-4 video file']
    CONSTRAINT [PK_MediaType] PRIMARY KEY  ([MediaTypeId])
)

CREATE TABLE [Playlist]
(
    [PlaylistId] INTEGER  NOT NULL, -- example: [1, 2, 3]
    [Name] NVARCHAR(120), -- example: ['Music', 'Movies', 'TV Shows']
    CONSTRAINT [PK_Playlist] PRIMARY KEY  ([PlaylistId])
)

CREATE TABLE [PlaylistTrack]
(
    [PlaylistId] INTEGER  NOT NULL, -- example: [1, 3, 5]
    [TrackId] INTEGER  NOT NULL, -- example: [1, 2, 3]
    CONSTRAINT [PK_PlaylistTrack] PRIMARY KEY  ([PlaylistId], [TrackId]),
    FOREIGN KEY ([PlaylistId]) REFERENCES [Playlist] ([PlaylistId]) 
		ON DELETE NO ACTION ON UPDATE NO ACTION,
    FOREIGN KEY ([TrackId]) REFERENCES [Track] ([TrackId]) 
		ON DELETE NO ACTION ON UPDATE NO ACTION
)

CREATE TABLE [Track]
(
    [TrackId] INTEGER  NOT NULL, -- example: [1, 6, 7]
    [Name] NVARCHAR(200)  NOT NULL, -- example: ['For Those About To Rock (We Salute You)', 'Balls to the Wall', 'Fast As a Shark']
    [AlbumId] INTEGER, -- example: [1, 2, 3]
    [MediaTypeId] INTEGER  NOT NULL, -- example: [1, 2, 3]
    [GenreId] INTEGER, -- example: [1, 2, 3]
    [Composer] NVARCHAR(220), -- example: ['Angus Young, Malcolm Young, Brian Johnson', 'F. Baltes, S. Kaufman, U. Dirkscneider & W. Hoffman', 'F. Baltes, R.A. Smith-Diesel, S. Kaufman, U. Dirkscneider & W. Hoffman']
    [Milliseconds] INTEGER  NOT NULL, -- example: [343719, 342562, 230619]
    [Bytes] INTEGER, -- example: [11170334, 5510424, 3990994]
    [UnitPrice] NUMERIC(10,2)  NOT NULL, -- example: [0.99, 1.99]
    CONSTRAINT [PK_Track] PRIMARY KEY  ([TrackId]),
    FOREIGN KEY ([AlbumId]) REFERENCES [Album] ([AlbumId]) 
		ON DELETE NO ACTION ON UPDATE NO ACTION,
    FOREIGN KEY ([GenreId]) REFERENCES [Genre] ([GenreId]) 
		ON DELETE NO ACTION ON UPDATE NO ACTION,
    FOREIGN KEY ([MediaTypeId]) REFERENCES [MediaType] ([MediaTypeId]) 
		ON DELETE NO ACTION ON UPDATE NO ACTION
)

