import kagglehub
import data_loading as dl
import time
import utils


sensing_streams = ["activity", "audio", "bt", "conversation", "dark", "gps", "phonecharge", "phonelock", "wifi", "wifi_location"]


data = dl.load_sensing_stream("gps", user_id=utils.format_user_id(0))
print(data)



# for stream in sensing_streams[3:]:
#     for user_id in range(60):
#         try:
#             data = dl.load_sensing_stream(stream, user_id=utils.format_user_id(user_id))
#             time.sleep(2)
#             # print(data.head())
#         except FileNotFoundError:
#             print(f"File not found for {stream} and user {user_id}")