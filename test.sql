CREATE SCHEMA train_flats_prices;
CREATE TABLE price.flats (
  id SERIAL PRIMARY KEY, 
  street VARCHAR(100) NOT NULL, 
  neighborhood VARCHAR(100) NOT NULL
);
INSERT INTO price.flats (id, neighborhood) 
VALUES 
  ('1', 'Somerset'), 
  ('2', 'Dorset');