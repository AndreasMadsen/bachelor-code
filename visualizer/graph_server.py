
import urllib.parse as url
import http.server
import os.path as path
import ujson as json
import numpy as np
import scipy.sparse

thisdir = path.dirname(path.realpath(__file__))


class GraphServer:
    def __init__(self, clusters, distance, connectivity, nodes, verbose=False):
        self._verbose = verbose
        if (self._verbose): print("Initializing graph server")

        self._clusters = clusters
        self._distance = scipy.sparse.csr_matrix(distance)
        self._connectivity = scipy.sparse.csr_matrix(connectivity)
        self._raw_nodes = nodes
        self._nodes = [[node['title'], node['website'], node['id']] for node in nodes]

        # Create a http server
        if (self._verbose): print("\tCreating http server")
        self._server = http.server.HTTPServer(('127.0.0.1', 8000), GraphServer.Responder)
        self._server._owner = self

    def listen(self):
        if (self._verbose): print("Server listening on http://127.0.0.1:8000")
        self._server.serve_forever()

    def fetch_article(self, id):
        return self._raw_nodes[id]

    def fetch_compare(self, a, b):
        return {
            "connecitivity": bool(self._connectivity[min(a, b), max(a, b)]),
            "distance": float(self._distance[min(a, b), max(a, b)])
        }

    def _groups_from_title(self, search):
        if (self._verbose): print("\tSearching for \"%s\"" % (search))
        words = search.split()

        # Construct and execute SQL search query
        match = np.fromiter([
            np.all([(word in node[0]) for word in words])
            for node in self._nodes
        ], dtype='bool')

        # Fetch groups
        groups = set(int(group) for group in self._clusters['node_to_group'][match])
        if (self._verbose): print("\tSearch complete, found %d groups" % len(groups))

        return groups

    def _fetch_single_group(self, group_id):
        if (self._verbose): print("\tFetching group %d" % group_id)

        # Create node info object
        nodes = self._clusters['group'][group_id, 0:self._clusters['group_size'][group_id]]
        node_info = [self._nodes[id] for id in nodes]

        # Create link info object
        if (self._verbose): print("\tBuilding link object")
        mask = np.any(self._clusters['connects_row'][:, np.newaxis] == nodes, axis=1)
        if (np.sum(mask) == 0):
            link_info = []
        else:
            info = (
                self._clusters['connects_row'][mask],
                self._clusters['connects_col'][mask]
            )

            link_info = [
                [int(row), int(col), float(data)]
                for (row, col, data)
                in zip(info[0], info[1], self._distance[info].A1)
            ]

        # Send group info
        return (node_info, link_info)

    def fetch_graph(self, groups):
        if (self._verbose): print("Fetching groups")

        # Validate groups
        max_group_size = int(self._clusters['group'].shape[0])
        for group in groups:
            if (group >= max_group_size):
                if (self._verbose): print("\tGroup with id %d do not exists" % group)
                return None

        # Initialize info array
        info = []

        # Fetch group info
        for group in groups:
            (node_info, link_info) = self._fetch_single_group(group)
            info.append({
                "group": group,
                "nodes": node_info,
                "links": link_info
            })

        # Done return result
        return info

    class Responder(http.server.BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self._owner = args[2]._owner
            self._verbose = self._owner._verbose
            http.server.BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

        def do_GET(self):
            if self.path         == '/'             : self.index_page()
            elif self.path       == '/details'      : self.details_page()
            elif self.path       == '/d3.js'        : self.d3_script()
            elif self.path       == '/view.js'      : self.view_script()
            elif self.path       == '/style.css'    : self.style_script()
            elif self.path[0:11] == '/graph.json'   : self.graph_data()
            elif self.path[0:13] == '/article.json' : self.article_data()
            elif self.path[0:13] == '/compare.json' : self.compare_data()
            else                                    : self.otherwise()

        def index_page(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=UTF-8')
            self.end_headers()

            f = open(path.join(thisdir, 'public', 'index.html'), 'rb')
            self.wfile.write(f.read())
            f.close()

        def details_page(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=UTF-8')
            self.end_headers()

            f = open(path.join(thisdir, 'public', 'details.html'), 'rb')
            self.wfile.write(f.read())
            f.close()

        def d3_script(self):
            self.send_response(200)
            self.send_header('Content-Type', 'application/javascript; charset=UTF-8')
            self.end_headers()

            f = open(path.join(thisdir, 'public', 'd3.js'), 'rb')
            self.wfile.write(f.read())
            f.close()

        def view_script(self):
            self.send_response(200)
            self.send_header('Content-Type', 'application/javascript; charset=UTF-8')
            self.end_headers()

            f = open(path.join(thisdir, 'public', 'view.js'), 'rb')
            self.wfile.write(f.read())
            f.close()

        def style_script(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/css; charset=UTF-8')
            self.end_headers()

            f = open(path.join(thisdir, 'public', 'style.css'), 'rb')
            self.wfile.write(f.read())
            f.close()

        def article_data(self):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=UTF-8')
            self.end_headers()

            query = url.parse_qs(url.urlparse(self.path).query)
            data = self._owner.fetch_article(int(query['id'][0]))
            self.wfile.write(bytes(json.dumps(data), 'ASCII'))

        def compare_data(self):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=UTF-8')
            self.end_headers()

            query = url.parse_qs(url.urlparse(self.path).query)
            data = self._owner.fetch_compare(int(query['a'][0]), int(query['b'][0]))
            self.wfile.write(bytes(json.dumps(data), 'ASCII'))

        def graph_data(self):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=UTF-8')
            self.end_headers()

            query = url.parse_qs(url.urlparse(self.path).query)

            # Convert input to group list
            if ('title' in query):
                groups = self._owner._groups_from_title(query['title'][0])
            elif ('groups' in query):
                groups = set(int(group) for group in query['groups'][0].split(","))
            else:
                groups = None

            # Fetch nodes and links
            if (groups is None):
                data = None
            else:
                data = self._owner.fetch_graph(groups)

            # Send data
            if (data is None):
                if (self._verbose): print("\tBad query input")
                self.wfile.write(bytes('null', 'ASCII'))
            else:
                if (self._verbose): print("\tSending result")

                self.wfile.write(bytes('[', 'ASCII'))
                for index, group in enumerate(data):
                    self.wfile.write(bytes('{"group":', 'ASCII'))
                    self.wfile.write(bytes(json.dumps(group['group']), 'ASCII'))

                    self.wfile.write(bytes(', "nodes":', 'ASCII'))
                    self.wfile.write(bytes(json.dumps(group['nodes']), 'ASCII'))

                    self.wfile.write(bytes(', "links":', 'ASCII'))
                    self.wfile.write(bytes(json.dumps(group['links']), 'ASCII'))

                    # Write } if last item otherwise write },\n
                    if (index == (len(data) - 1)):
                        self.wfile.write(bytes('}', 'ASCII'))
                    else:
                        self.wfile.write(bytes('},\n', 'ASCII'))
                self.wfile.write(bytes(']', 'ASCII'))

                if (self._verbose): print("\tResult send")

        def otherwise(self):
            self.send_response(404)
            self.send_header('Content-Type', 'text/html; charset=UTF-8')
            self.end_headers()

            self.wfile.write(bytes('<pre>Sorry invalid path (404)</pre>', 'UTF-8'))
