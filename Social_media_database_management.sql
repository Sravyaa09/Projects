#creating table for bookmarks
CREATE TABLE bookmarks (
	bookmarks_id int,
    post_id int(100),
    user_id int(100)
);

#creating table for comments_likes
CREATE TABLE comments_likes (
	comment_like_id int,
    user_id int(100),
    comment_id int(100)
);

#creating table for comments
CREATE TABLE comments (
	comments_id int,
    comment_text VARCHAR(255),
    post_id int(100),
    user_id int(100)
);

#creating table for follows
CREATE TABLE Follows (
	follows_id int,
    followers_id int(100),
    followees_id int(100)
);

#creating table for hashtag_follow
CREATE TABLE hashtag_follow (
	hashtag_follows_id int,
    user_id int(100),
    hashtag_id int(100)
);

#creating table for follows
CREATE TABLE hashtag (
	hashtag_id int,
    hashtag_name VARCHAR(255)
);

#creating table for login
CREATE TABLE login (
	login_id int,
    user_id int(100),
    ip int(100)
);

#creating table for photos
CREATE TABLE photos (
	photos_id int,
    photo_url VARCHAR(255),
    post_id int(100),
    size int(100)
);

#creating table for posts
CREATE TABLE posts (
	post_id int,
    photo_id int(100),
    video_id int(100),
    user_id int(100),
    caption VARCHAR(255),
    location VARCHAR(255)
);

#creating table for post_likes
CREATE TABLE post_likes (
	post_likes_id int,
    user_id int(100),
    post_id int(100)
);

#creating table for post_tags
CREATE TABLE post_likes (
	post_tags_id int,
    post_id int(100),
    hashtag_id int(100)
);

#creating table for users
CREATE TABLE users (
	user_id int,
    username VARCHAR(255),
    profile_photo_URL VARCHAR(255),
    bio VARCHAR(255),
    email VARCHAR(255)
);

