from song_detection.custom_detect import CustomDetect
from typing import Optional

class SongDetectGroup:
    def __init__(self, title: Optional[CustomDetect], data: Optional[CustomDetect], sheet: Optional[CustomDetect]):
        self.title = title
        self.data = data
        self.sheet = sheet

        # set image to the first non-null object
        for obj in (title, data, sheet):
            if obj:
                self.image = obj.image
                break

def filterDuplicates(detects: list[CustomDetect]) -> list[CustomDetect]:
    # filter duplicates in detects. If there are two detects with the same label and one of them is inside the other, remove the smaller one

    sortedDetects = sorted(detects, key=lambda x: x.bounds.area(), reverse=True)
    resultArray = sortedDetects.copy()

    for i in range(len(sortedDetects)):
        for j in range(i + 1, len(sortedDetects)):
            if sortedDetects[i].label == sortedDetects[j].label and sortedDetects[i].bounds.isInside(sortedDetects[j].bounds):
                resultArray.remove(sortedDetects[j])

    return resultArray


def groupCustomDetect(detects: list[CustomDetect]) -> list[SongDetectGroup]:

    # filter duplicates in detects. If there are two detects with the same label and one of them is inside the other, remove the smaller one
    detects = filterDuplicates(detects)

    # start grouping
    ungroupedDetects = detects.copy()
    # loop through all sheets and find the title and data which are inside the sheet
    # then remove them from the ungroupedDetects list
    # then add the group to the groups list

    groups : list[SongDetectGroup] = []

    sheets = list(filter(lambda x: x.label == "sheet", ungroupedDetects))
    for sheet in sheets:
        title = next((x for x in ungroupedDetects if x.label == "title" and x.bounds.isInside(sheet.bounds, True)), None)
        data = next((x for x in ungroupedDetects if x.label == "data" and x.bounds.isInside(sheet.bounds, True)), None)

        groups.append(SongDetectGroup(title, data, sheet))
        if title: 
            ungroupedDetects.remove(title)
        if data:
            ungroupedDetects.remove(data)
        ungroupedDetects.remove(sheet)

    # loop over all ungrouped detects with label data and find the closest title
    # then add the group to the groups list
    for data in list(filter(lambda x: x.label == "data", ungroupedDetects)):
        # find the closest title to the data with left and top coordinates
        closestTitle = None
        closestDistance = float("inf")
        for title in list(filter(lambda x: x.label == "title", ungroupedDetects)):
            distance = (title.bounds.left - data.bounds.left) ** 2 + (title.bounds.top - data.bounds.top) ** 2
            if distance < closestDistance:
                closestDistance = distance
                closestTitle = title

        groups.append(SongDetectGroup(closestTitle, data, None))
        if closestTitle:
            ungroupedDetects.remove(closestTitle)
        ungroupedDetects.remove(data)

    # loop over all ungrouped detects with label title
    # then add the group to the groups list
    for title in list(filter(lambda x: x.label == "title", ungroupedDetects)):
        groups.append(SongDetectGroup(title, None, None))
        ungroupedDetects.remove(title)

    return groups