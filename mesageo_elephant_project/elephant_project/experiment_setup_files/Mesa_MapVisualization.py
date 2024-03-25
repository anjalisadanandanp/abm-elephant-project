from mesa_geo.visualization.MapModule import MapModule

"""
Overwriting function:render in MapModule

Portrayal of Human agent, Elephant agents and food 

"""

class MapModule(MapModule):
    """A MapModule for Leaflet maps.""" 

    def render(self, model):

        featurecollection = dict(type="FeatureCollection", features=[])

        for agent in model.grid.agents:

            agent_vars = agent.__geo_interface__()   #agent_vars is a dictionary: (['type', 'geometry', 'properties'])
            portrayal = self.portrayal_method(agent)    #The portrayal dict defined in server.py
            #print(portrayal)

            agent_portrayal = {}
            agent_portrayal["type"] = agent_vars["type"]
            agent_portrayal["geometry"] = agent_vars["geometry"]
            agent_portrayal["properties"] = {}      #we are overwriting properties for rendering the agents

            #print(agent_vars["properties"].keys())

            for key, value in portrayal.items():
                agent_portrayal["properties"][key] = value

            agent_visibility_portrayal = {}     #For rendering the visibility range of the agents
            agent_visibility_portrayal["type"] = agent_vars["type"]
            agent_visibility_portrayal["geometry"] = agent_vars["geometry"]
            agent_visibility_portrayal["properties"] = {}
            # agent_visibility_portrayal["properties"]["color"] = "yellow"
            agent_visibility_portrayal["properties"]["category"] = "visibility"
            agent_visibility_portrayal["properties"]["layer"] = 0
            agent_visibility_portrayal["properties"]["filled"] = True
            agent_visibility_portrayal["properties"]["fitness"] = ""
            agent_visibility_portrayal["properties"]["aggression"] = ""


            if "bull" in agent.unique_id:
                agent_portrayal["properties"]["category"] = "Elephant"
                agent_portrayal["properties"]["color"] = "red"
                agent_visibility_portrayal["properties"]["radius"] = 5
                agent_visibility_portrayal["properties"]["color"] = "red"

            else:
                agent_portrayal["properties"]["category"] = "Human"
                agent_portrayal["properties"]["color"] = "blue"
                agent_visibility_portrayal["properties"]["radius"] = 1
                agent_visibility_portrayal["properties"]["color"] = "blue"

            if agent.fitness <= 0:
                agent_portrayal["properties"]["color"] = "grey"
                agent_visibility_portrayal["properties"]["radius"]= 1
                agent_visibility_portrayal["properties"]["color"]= "grey"

            featurecollection["features"].append(agent_portrayal)
            featurecollection["features"].append(agent_visibility_portrayal) 

        return featurecollection
