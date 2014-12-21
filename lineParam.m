files = {'line.txt'};

for i=1:1:4
    [k,b,theta,rho] = textread(files{i});
     figure(i);
     subplot(2,1,1); plot(k,b,'o');
     xlabel('k');
     ylabel('b');
     title(files{i});
     axis([-15,10,-1000,3000]);
     grid on
     
     subplot(2,1,2); plot(theta,rho,'o');
     xlabel('theta');
     ylabel('rho');
     axis([0,3,0,300]);
     grid on
end