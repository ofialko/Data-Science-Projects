
import tweepy

# database interface
import sqlite3
conn = sqlite3.connect('tweets.db')
curs = conn.cursor()
 
class My_Stream(tweepy.StreamListener):
    """ Handles all incoming tweets as discrete tweet objects.
    """
 
    def on_status(self, status):
        """Called when status (tweet) object received.
        """
        try:
            tid = status.id_str
            usr = status.author.screen_name
            lang = status.lang
            txt = status.text
            location = status.user.location
            src = status.source
            cat = status.created_at
 
            # Now that we have our tweet information, let's stow it away in our 
            # sqlite database
            curs.execute("insert into tweets (tid,username, created_at, lang, content, location, source)\
                values(?,?,?,?,?,?,?)", (tid,usr, cat, lang, txt, location, src))  
            conn.commit()
        except Exception as e:
            # Most errors we're going to see relate to the handling of UTF-8 messages (sorry)
            print(e)
 
    def on_error(self, status_code):
       print('Error! Status code = %s' % status_code)
       return True
 
def main():
    # establish stream
    consumer_key = 'Zba2dlLdOVKDi03NjMje9UqB8'
    consumer_secret = 'UapRStoiNiufXlht2BKHotxl1ZDvaGLCkrNjebxQjCUyTcMJc5'
    access_token = '4919141599-RYUkTG1vLYcHU62dDUgODYnL48xQzXT05tV4M3z'
    access_token_secret = 'eun0n1em96zTfT0QF4tQ2Vpu6AFxrVYGZr7D09CiRhrD9'

    auth1 = tweepy.auth.OAuthHandler(consumer_key, consumer_secret)
    auth1.set_access_token(access_token, access_token_secret)
 
    
    while True:
        try:
                stream = tweepy.Stream(auth1, My_Stream(), timeout=None)
                print("Stream established")
                keywords = ['Pokemon']
                stream.filter(track=keywords) #,languages=["en"])            
        except KeyboardInterrupt: 
                print("Disconnecting from database...")
                conn.commit()
                conn.close()
                print("Done")
                return
        except:
                continue
         
    

if __name__ == '__main__':
    main()
   