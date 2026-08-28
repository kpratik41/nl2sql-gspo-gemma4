CREATE TABLE "customers" (
"index" INTEGER, -- example: [0, 1, 2]
  "customer_id" TEXT, -- example: ['06b8999e2fba1a1fbc88172c00ba8bc7', '18955e83d337fd6b2def6b18a428ac77', '4e7b3e00288586ebd08712fdd0374a03']
  "customer_unique_id" TEXT, -- example: ['861eff4711a542e4b93843c6dd7febb0', '290c77bc529b7ac935b93aa66c333dc3', '060e732b5b29e8181a18229c7b0b2b5e']
  "customer_zip_code_prefix" INTEGER, -- example: [14409, 9790, 1151]
  "customer_city" TEXT, -- example: ['franca', 'sao bernardo do campo', 'sao paulo']
  "customer_state" TEXT -- example: ['SP', 'SC', 'MG']
)

CREATE TABLE "sellers" (
"index" INTEGER, -- example: [0, 1, 2]
  "seller_id" TEXT, -- example: ['3442f8959a84dea7ee197c632cb2df15', 'd1b65fc7debc3361ea86b5f14c68d2e2', 'ce3ad9de960102d0677a81f5d0bb7b2d']
  "seller_zip_code_prefix" INTEGER, -- example: [13023, 13844, 20031]
  "seller_city" TEXT, -- example: ['campinas', 'mogi guacu', 'rio de janeiro']
  "seller_state" TEXT -- example: ['SP', 'RJ', 'PE']
)

CREATE TABLE "order_reviews" (
"index" INTEGER, -- example: [0, 1, 2]
  "review_id" TEXT, -- example: ['7bc2406110b926393aa56f80a40eba40', '80e641a11e56f04c1ad469d5645fdfde', '228ce5500dc1d8e020d8d1322874b6f0']
  "order_id" TEXT, -- example: ['73fc7af87114b39712e6da79b0a377eb', 'a548910a1c6147796b98fdf73dbeba33', 'f9e4b658b201a9f2ecdecbb34bed034b']
  "review_score" INTEGER, -- example: [4, 5, 1]
  "review_comment_title" TEXT, -- example: ['recomendo', 'Super recomendo', 'Não chegou meu produto ']
  "review_comment_message" TEXT, -- example (truncated): ['Recebi bem antes do prazo estipulado.', 'Parabéns lojas lannister adorei comprar pela Internet seguro e prático Parabéns a todos feliz Páscoa', 'aparelho eficiente. no site a marca do aparelho esta impresso como 3desinfector e ao chegar esta com...']
  "review_creation_date" TEXT, -- example: ['2018-01-18 00:00:00', '2018-03-10 00:00:00', '2018-02-17 00:00:00']
  "review_answer_timestamp" TEXT -- example: ['2018-01-18 21:46:59', '2018-03-11 03:05:13', '2018-02-18 14:36:24']
)

CREATE TABLE "order_items" (
"index" INTEGER, -- example: [0, 1, 2]
  "order_id" TEXT, -- example: ['00010242fe8c5a6d1ba2dd792cb16214', '00018f77f2f0320c557190d7a144bdd3', '000229ec398224ef6ca0657da4fc703e']
  "order_item_id" INTEGER, -- example: [1, 2, 3]
  "product_id" TEXT, -- example: ['4244733e06e7ecb4970a6e2683c13e61', 'e5f2d52b802189ee658865ca93d83a8f', 'c777355d18b72b67abbeef9df44fd0fd']
  "seller_id" TEXT, -- example: ['48436dade18ac8b2bce089ec2a041202', 'dd7ddc04e1b6c2c614352b383efe2d36', '5b51032eddd242adc84c38acab88f23d']
  "shipping_limit_date" TEXT, -- example: ['2017-09-19 09:45:35', '2017-05-03 11:05:13', '2018-01-18 14:48:30']
  "price" REAL, -- example: [58.9, 239.9, 199.0]
  "freight_value" REAL -- example: [13.29, 19.93, 17.87]
)

CREATE TABLE "products" (
"index" INTEGER, -- example: [0, 1, 2]
  "product_id" TEXT, -- example: ['1e9e8ef04dbcff4541ed26657ea517e5', '3aa071139cb16b67ca9e5dea641aaa2f', '96bd76ec8810374ed1b65e291975717f']
  "product_category_name" TEXT, -- example: ['perfumaria', 'artes', 'esporte_lazer']
  "product_name_lenght" REAL, -- example: [40.0, 44.0, 46.0]
  "product_description_lenght" REAL, -- example: [287.0, 276.0, 250.0]
  "product_photos_qty" REAL, -- example: [1.0, 4.0, 2.0]
  "product_weight_g" REAL, -- example: [225.0, 1000.0, 154.0]
  "product_length_cm" REAL, -- example: [16.0, 30.0, 18.0]
  "product_height_cm" REAL, -- example: [10.0, 18.0, 9.0]
  "product_width_cm" REAL -- example: [14.0, 20.0, 15.0]
)

CREATE TABLE "geolocation" (
"index" INTEGER, -- example: [0, 1, 2]
  "geolocation_zip_code_prefix" INTEGER, -- example: [1037, 1046, 1041]
  "geolocation_lat" REAL, -- example: [-23.54562128115268, -23.54608112703553, -23.54612896641469]
  "geolocation_lng" REAL, -- example: [-46.63929204800168, -46.64482029837157, -46.64295148361138]
  "geolocation_city" TEXT, -- example: ['sao paulo', 'são paulo', 'sao bernardo do campo']
  "geolocation_state" TEXT -- example: ['SP', 'RN', 'AC']
)

CREATE TABLE "product_category_name_translation" (
"index" INTEGER, -- example: [0, 1, 2]
  "product_category_name" TEXT, -- example: ['beleza_saude', 'informatica_acessorios', 'automotivo']
  "product_category_name_english" TEXT -- example: ['health_beauty', 'computers_accessories', 'auto']
)

CREATE TABLE "orders" (
"index" INTEGER, -- example: [0, 1, 2]
  "order_id" TEXT, -- example: ['e481f51cbdc54678b7cc49136f2d6af7', '53cdb2fc8bc7dce0b6741e2150273451', '47770eb9100c2d0c44946d9cf07ec65d']
  "customer_id" TEXT, -- example: ['9ef432eb6251297304e76186b10a928d', 'b0830fb4747a6c6d20dea0b8c802d7ef', '41ce2a54c0b03bf3443c3d931a367089']
  "order_status" TEXT, -- example: ['delivered', 'invoiced', 'shipped']
  "order_purchase_timestamp" TEXT, -- example: ['2017-10-02 10:56:33', '2018-07-24 20:41:37', '2018-08-08 08:38:49']
  "order_approved_at" TEXT, -- example: ['2017-10-02 11:07:15', '2018-07-26 03:24:27', '2018-08-08 08:55:23']
  "order_delivered_carrier_date" TEXT, -- example: ['2017-10-04 19:55:00', '2018-07-26 14:31:00', '2018-08-08 13:50:00']
  "order_delivered_customer_date" TEXT, -- example: ['2017-10-10 21:25:13', '2018-08-07 15:27:45', '2018-08-17 18:06:29']
  "order_estimated_delivery_date" TEXT -- example: ['2017-10-18 00:00:00', '2018-08-13 00:00:00', '2018-09-04 00:00:00']
)

CREATE TABLE "order_payments" (
"index" INTEGER, -- example: [0, 1, 2]
  "order_id" TEXT, -- example: ['b81ef226f3fe1789b1e8b2acac839d17', 'a9810da82917af2d9aefd1278f1dcfa0', '25e8ea4e93396b6fa0d3dd708e76c1bd']
  "payment_sequential" INTEGER, -- example: [1, 2, 4]
  "payment_type" TEXT, -- example: ['credit_card', 'boleto', 'voucher']
  "payment_installments" INTEGER, -- example: [8, 1, 2]
  "payment_value" REAL -- example: [99.33, 24.39, 65.71]
)

