import utils

print("Welcome to NewsNow!")
choice = 'y'
while choice.lower() == 'y':
    query = input("Enter query to search: ").lower()
    links = utils.getLinks(query, 5)
    articles = utils.getDocuments(links)
    merged_article = utils.merge(articles)
    summary = utils.summarize(merged_article)
    print(f"Latest on {query}:\n{summary}\n")
    choice = input("Search for another article? [y/n]: ")
