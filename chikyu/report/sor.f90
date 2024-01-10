program sor
  implicit none
  real, dimension(4,4) :: a
  real, dimension(4,1) :: b, x, copy_x
  real :: dx, w, difference, x_childa
  integer ::n=4, i, j, k=0
  a = reshape ( (/1,-1,0,0,-1,2,-1,0,0,-1,2,-1,0,0,-1,2/), (/4,4/) )
  b = reshape ( (/-1, 0, 0, 0/), (/4,1/) )


  difference = 1
  dx = 0.001
  k = 0

  do j = 1, 9, 1
    w = j/10.
    x = 0.0
    copy_x = x

    do while (difference > dx)
      k = k + 1 !k is the number of iterations

      do i = 1, n
        x_childa = ( b(i,1) - sum(a(:i-1,i)*x(:i-1,1)) - sum(a(i+1:n,i)*x(i+1:n,1)) )/a(i,i)
        x(i,1) = (1-w)*x(i,1) + w*x_childa
      end do

      difference = (sum((x - copy_x)**2))**0.5
      print *, j, k, difference
      copy_x = x

    end do

    k = 0 ! reset 
    print *, x
    
  end do

end program sor

