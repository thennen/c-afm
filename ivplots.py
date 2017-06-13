# df = pd.read_pickle whatever

iv = df[df.type == 'iv']
riv = iv[iv.folder == '09-Jun-2017'].sort('ctime')

fig, ax = plt.subplots()
colors = iter(['#F60202', '#019494', '#F67102', '#02C502'])
labels = iter(['Conducting region before','Insulating region before','Conducting region after','Insulating region after'])
for loc, group in riv.groupby('x_offset'):
    color = next(colors)
    for row, iv in group.iterrows():
        plot(iv.V, iv.I * 1e9, color=color)
        #plot(iv.V2, iv.I2 * 1e9, color=color)
    ax.lines[-1].set_label(next(labels))
legend()
ylabel('Tip current [nA]')
xlabel('Applied Voltage [V]')

# IV loops are easy to average because they have the same x coordinates ..
def meanloop(group):
    # take average of forward and backward loops
    # Wait there are no backward loops this time..
    I1 = group.I.mean()
    V1 = group.V.mean()
    #I2 = group.I2.mean()
    #V2 = group.V2.mean()
    #meanI = np.mean(I1, I2)
    #meanV = np.mean(V1, V2)
    return pd.Series(dict(I = I1, V=V1))
iv_avg = riv.groupby('x_offset').apply(meanloop)

fig, ax = plt.subplots()
colors = iter(['#F60202', '#019494', '#F67102', '#02C502'])
labels = iter(['Conducting region before','Insulating region before','Conducting region after','Insulating region after'])
for loc, loop in iv_avg.iterrows():
    color = next(colors)
    plot(loop.V, loop.I * 1e9, color=color)
    ax.lines[-1].set_label(next(labels))
legend()
ylabel('Tip current [nA]')
xlabel('Applied Voltage [V]')

