import json
import os
import re

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from pyyoutube import Api
from youtube_transcript_api import YouTubeTranscriptApi

def scrapePlaylistTranscripts(api, channel_username, start_playlist = 0, end_playlist = 1000, count_playlists = 1000, count_videos = 200):
	channel = api.get_channel_info(for_username = channel_username)
	playlists = api.get_playlists(channel_id = channel.items[0].id, count = count_playlists)

	output_json = {
		"playlists": []
	}

	for i, playlist in enumerate(playlists.items):
		if i < start_playlist or i >= end_playlist:
			print("Continue")
			continue

		print(i, playlist.snippet.title)

		output_json["playlists"].append({
			"id": playlist.id,
			"title": playlist.snippet.title,
			"videos": []
		})

		playlist_items = api.get_playlist_items(playlist_id = playlist.id, count = count_videos)

		for playlist_item in playlist_items.items:
			video_id = playlist_item.snippet.resourceId.videoId

			try:
				video = api.get_video_by_id(video_id = video_id).items[0]
			except:
				print("Video request failed, putting in empty record")
				output_json["playlists"][-1]["videos"].append({
					"id": video_id,
					"title": None,
					"transcript": None
				})

				continue

			video_title = video.snippet.title

			print("\tScraping video: ", video_id, video_title)

			try:
				transcript = YouTubeTranscriptApi.get_transcript(video_id)
			except:
				# Transcript unavailable
				transcript = None

			output_json["playlists"][-1]["videos"].append({
				"id": video_id,
				"title": video_title,
				"transcript": transcript
			})

	return output_json

def extractKeywords(transcripts):
	keywords_json = {
		"playlists": []
	}

	for i, playlist in enumerate(transcripts["playlists"]):
		keywords_json["playlists"].append({
			"id": playlist["id"],
			"title": playlist["title"],
			"videos": []
		})

		for j, video in enumerate(playlist["videos"]):
			print("\tExtracting keywords video: ", video["id"], video["title"])

			if video["transcript"] is None:
				video_keywords = None
			else:
				video_keywords = extractVideoKeywords(video)

			keywords_json["playlists"][-1]["videos"].append({
				"id": video["id"],
				"title": video["title"],
				"keywords": video_keywords
			})

	return keywords_json

def extractVideoKeywords(video):
    video_text = " ".join(transcript["text"] for transcript in video["transcript"])

    resp = chat([SystemMessage(content = "You extract the main keywords in the text and extract these into a comma separated list. Please prefix the keywords with 'Keywords:'"),
                HumanMessage(content = video_text)])
    answer = resp.content

    answer = answer.lower()
    expression = r".*keywords:(.+?)$"

    if re.search(expression, answer):
        keywords = re.sub(expression, r"\1", text, flags = re.S)
        if keywords is not None and len(keywords) > 0:
            return [re.sub(r"\.$", "", k.strip()) for k in keywords.strip().split(',')]

    return []

if __name__ == "__main__":
	with open("yt_api_key.txt", "r") as fp:
		YT_API_KEY = fp.readline()

	with open("openai_api_key.txt", "r") as fp:
		OPENAI_API_KEY = fp.readline()

	os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

	api = Api(api_key = YT_API_KEY)
	# chat = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0)

	khanacademyTranscripts = scrapePlaylistTranscripts(api, channel_username = "khanacademy", start_playlist = 0, end_playlist = 600, count_playlists = 600, count_videos = 300)

	with open("transcripts/khanacademy.json", "w") as fp:
		json.dump(khanacademyTranscripts, fp, indent = 4)

	crashcourseTranscripts = scrapePlaylistTranscripts(api, channel_username = "crashcourse", start_playlist = 0, end_playlist = 60, count_playlists = 60, count_videos = 100)

	with open("transcripts/crashcourse.json", "w") as fp:
		json.dump(crashcourseTranscripts, fp, indent = 4)

	# khanacademyKeywords = extractKeywords(khanacademyTranscripts)

	# with open("keywords/khanacademy.json", "w") as fp:
	# 	json.dump(khanacademyKeywords, fp, indent = 4)