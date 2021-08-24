import numpy as np

import plotly.graph_objects as go # 시각화 라이브러리
import plotly.express as px




def read_off(file):
    off_header = file.readline().strip() # file을 읽어오기. 이 때 맨 첫줄에는 off형식일 경우 OFF로 시작한다.
    if 'OFF' == off_header:
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')]) # 맨 첫줄은 갯수에 대한 정보가 들어있다.
    else:
        n_verts, n_faces, __ = tuple([int(s) for s in off_header[3:].split(' ')]) # 맨 첫줄이 개행이 안된 상태라면 OFF 다음걸로 가져온다.
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)] # 이제 이 아래로 vertex 행 개수만큼 가져오면 된다. x, y, z
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)] # 이 다음은 faces, (표면)
    return verts, faces

def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
        layout=go.Layout(
            updatemenus=[dict(type='buttons',
                showactive=False,
                y=1,
                x=0.8,
                xanchor='left',
                yanchor='bottom',
                pad=dict(t=45, r=10),
                buttons=[dict(label='Play',
                    method='animate',
                    args=[None, dict(frame=dict(duration=50, redraw=True),
                        transition=dict(duration=0),
                        fromcurrent=True,
                        mode='immediate'
                        )]
                    )
                ])]
        ),
        frames=frames
    )

    return fig

def pcshow(xs,ys,zs):
    data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                   mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()

