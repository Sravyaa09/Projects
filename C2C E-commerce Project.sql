SELECT DISTINCT b.buyers, u.country, u.hasAndroidApp, u.hasIosApp FROM `c2c-fashion-store`.`buyers-repartition-by-country` b
INNER JOIN `c2c-fashion-store`.`6m-0k-99k.users.dataset.public` u ON b.`country` = u.`country`
WHERE u.hasAndroidApp = 'True' and u.hasIosApp = 'True';

SELECT DISTINCT(b.country), SUM(b.buyers) as buyers, SUM(CASE WHEN u.hasProfilePicture = 'TRUE' THEN 1 ELSE 0 END) as hasProfilePicture, SUM(b.totalproductsbought) as totalproductsbought, SUM(b.totalproductswished) as totalproductswished
FROM `c2c-fashion-store`.`buyers-repartition-by-country` b
INNER JOIN `c2c-fashion-store`.`6m-0k-99k.users.dataset.public` u ON b.`country` = u.`country`
GROUP BY b.country
ORDER BY b.country;

SELECT DISTINCT(s.country), SUM(s.nbsellers) as TotalFollowers, ROUND(AVG(u.productsSold), 2) as ProductsSold, ROUND(AVG(u.productsPassRate), 2) as ProductPassRate, ROUND(AVG(s.percentofappusers), 2) as PercentOfAppUsers, ROUND(AVG(s.percentofiosusers), 2) as PercentOfIOSUsers
FROM `c2c-fashion-store`.`Comparison-of-Sellers-by-Gender-and-Country` s
INNER JOIN `c2c-fashion-store`.`6m-0k-99k.users.dataset.public` u ON s.`country` = u.`country`
GROUP BY s.country
ORDER BY s.country;

#Compare the average number of products liked by users with and without a profile picture in each country
SELECT country, 
       AVG(CASE WHEN hasProfilePicture THEN socialProductsLiked ELSE NULL END) AS avg_liked_with_pic,
       AVG(CASE WHEN NOT hasProfilePicture THEN socialProductsLiked ELSE NULL END) AS avg_liked_without_pic
FROM `c2c-fashion-store`.`6m-0k-99k.users.dataset.public`
GROUP BY country;

#Identify countries with the highest variance in the number of products sold by sellers
SELECT country, VARIANCE(totalproductssold) AS variance_products_sold
FROM `c2c-fashion-store`.`Comparison-of-Sellers-by-Gender-and-Country`
GROUP BY country
ORDER BY variance_products_sold DESC
LIMIT 5;

#Find the ratio of female to male buyers and their respective buying ratios for each country
SELECT t1.country, t1.femalebuyers / t1.malebuyers AS female_to_male_ratio, 
       t1.femalebuyersratio, t1.topfemalebuyersratio
FROM `c2c-fashion-store`.`buyers-repartition-by-country` t1;







