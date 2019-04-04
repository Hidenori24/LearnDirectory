plot(time,Vec1,type="o",xlab="",ylab="loss",ylim=c(0,0.5))
points(time,Vec2,type="o",pch=16)
legend(10,0.5,legend=c("Non Use CNN","CNN"),pch=c(1,16))
